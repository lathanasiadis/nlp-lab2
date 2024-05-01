import os
import warnings
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt 

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN, LSTM
from attention import SimpleSelfAttentionModel
from early_stopper import EarlyStopper
from training import train_dataset, eval_dataset, get_metrics_report, torch_train_val_split
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

def train_model(model, train_loader, val_loader, criterion, optimizer, print_freq=None, use_lens=True):
    train_losses = []
    val_losses = []
    was_early_stop = False

    for epoch in range(1, EPOCHS + 1):
        # train the model for one epoch
        train_dataset(epoch, train_loader, model, criterion, optimizer, use_lens)

        # evaluate the performance of the model, on both data sets
        train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                                model,
                                                                criterion,
                                                                use_lens)
        val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader,
                                                          model,
                                                          criterion,
                                                          use_lens)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if print_freq is not None and epoch % print_freq == 0:
            print(f"Epoch: {epoch}")
            print(f"Train Loss: {train_loss}")
            print(f"Validation Loss: {val_loss}\n")
            
        if stopper.early_stop(val_loss):
            print("Early stop!")
            was_early_stop = True
            break
    
    return train_losses, val_losses, was_early_stop

def plot_loss_curves(train_losses, val_losses, was_early_stop, model_name=None):
    i = len(train_losses)
    x_axis = range(1, i+1)
    plt.plot(x_axis, train_losses, label="Train set")
    plt.plot(x_axis, val_losses, label="Validation set")
    plt.xticks(x_axis)
    if was_early_stop:
        plt.axvline(i - PATIENCE, linestyle="--",
                    label="Early Stop", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # control maximum number of ticks on x axis
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=18))
    if model_name is not None:
        plt.savefig("graph_{}.png".format(model_name))
    plt.show()

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",
    choices=["dnn", "lstm", "bilstm", "simple-tf", "multi-tf"],
    help="""Model used to predict sentiment. Available options:
    dnn for a Feed Forward NN with one hidden layer
    lstm for an LSTM"
    bilstm for a bidirectional LSTM
    simple-tf for a simple Self Attention Model
    multi-tf for a multi-head Self Attention Model""",
    default="simple-tf"
)
parser.add_argument("-d", "--dataset",
    choices=["mr", "semeval"],
    help="""Dataset used to train and evaluate the model. Available options:
    mr for Sentence Polarity Dataset [default]
    semeval for Semeval 2017 Task4-A""",
    default="mr"
)
parser.add_argument("-v", "--verbose", action="store_true",
    help="If set, will print some sample label encodings, sentence encodings, etc")

args = parser.parse_args()

DATASET = "MR" if args.dataset == "mr" else "Semeval2017A"
model_name = args.model
MODEL_PATH = "model_{}.pt".format(model_name)
SHOULD_PRINT_PRE = args.verbose

########################################################
# Configuration
########################################################

# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")
# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50
EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Early stop parameters
PATIENCE = 5

########################################################
# Define PyTorch datasets and dataloaders
########################################################

print("Will train a {} model on {}".format(model_name, DATASET))

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

le = LabelEncoder()
le.fit(y_train)

y_train = le.transform(y_train)  # EX1
y_test = le.transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size

# EX1: Print some sample encodings
sample_classes = le.inverse_transform(y_train[:10])
if SHOULD_PRINT_PRE:
    print("Encoded {} classes".format(n_classes))
    for i in range(10):
        print("{} -> {}".format(sample_classes[i], y_train[i]))

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx,
                            tweets=(DATASET == "Semeval2017A"), verbose=SHOULD_PRINT_PRE)
if SHOULD_PRINT_PRE:
    for i in range(5):
        print("Example #{}".format(i+1))
        print("{}\n{}".format(X_train[i], train_set[i]))

test_set = SentenceDataset(X_test, y_test, word2idx,
                           tweets=(DATASET == "Semeval2017A"), verbose=SHOULD_PRINT_PRE, max_len=train_set.max_len)

# EX7 - Define our PyTorch-based DataLoader
# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # EX7
train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE) # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################


if model_name == "dnn":
    # EX8
    model = BaselineDNN(output_size=n_classes,
                        embeddings=embeddings,
                        trainable_emb=EMB_TRAINABLE)
elif model_name == "lstm":
    model = LSTM(output_size=n_classes,
                 embeddings=embeddings,
                 trainable_emb=EMB_TRAINABLE,
                 bidirectional=False)
elif model_name == "bilstm":
    model = LSTM(output_size=n_classes,
                 embeddings=embeddings,
                 trainable_emb=EMB_TRAINABLE,
                 bidirectional=True)
elif model_name == "simple-tf":
    model = SimpleSelfAttentionModel(n_classes, embeddings, train_set.max_len)
elif model_name == "multi-tf":
    raise NotImplementedError

stopper = EarlyStopper(model, MODEL_PATH, PATIENCE, min_delta=1e-4)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = nn.CrossEntropyLoss()  # EX8
parameters = [param for param in model.parameters() if param.requires_grad]  # EX8
optimizer = torch.optim.Adam(parameters)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
model_is_tf = (model_name == "simple-tf") or (model_name == "multi-tf")
use_lens = not model_is_tf

train_losses, val_losses, was_early_stop = train_model(
      model, train_loader, val_loader, criterion, optimizer, print_freq=None, use_lens=use_lens)

plot_loss_curves(train_losses, val_losses, was_early_stop, model_name)

if was_early_stop:
    best = torch.load(MODEL_PATH)
    model.load_state_dict(best)
    
test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                     model,
                                                     criterion,
                                                     use_lens)

print("Classification report (test set)")
print(get_metrics_report(y_test_gold, y_test_pred))
