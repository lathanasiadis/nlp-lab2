import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt 

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from early_stopper import EarlyStopper
from training import train_dataset, eval_dataset, get_metrics_report, torch_train_val_split
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

from IPython import embed

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "model_dnn.pt"
PATIENCE = 5

SHOULD_PRINT_PRE = False

########################################################
# Define PyTorch datasets and dataloaders
########################################################

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
                           tweets=(DATASET == "Semeval2017A"), verbose=SHOULD_PRINT_PRE)

# EX7 - Define our PyTorch-based DataLoader
# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # EX7
train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE) # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = BaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

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

train_losses = []
val_losses = []
test_losses = []
was_early_stop = False

for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)
    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader,
                                                      model,
                                                      criterion)
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)

    if stopper.early_stop(val_loss):
        print("Early stop!")
        was_early_stop = True
        break


def plot_loss_curves(train_losses, val_losses, test_losses, save_path=None):
    i = len(train_losses)
    x_axis = range(1, i+1)
    plt.plot(x_axis, train_losses, label="Train set")
    plt.plot(x_axis, val_losses, label="Validation set")
    plt.plot(x_axis, test_losses, label="Test set")
    plt.xticks(x_axis)
    if was_early_stop:
        plt.axvline(i - PATIENCE, linestyle="--",
                    label="Early Stop", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # control maximum number of ticks on x axis
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=18))
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


plot_loss_curves(train_losses, val_losses, test_losses)

    
if was_early_stop:
    best = torch.load(MODEL_PATH)
    model.load_state_dict(best)
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)
    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader,
                                                      model,
                                                      criterion)
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)

print("Classification report (train set)")
print(get_metrics_report(y_train_gold, y_train_pred))
print("Classification report (val set)")
print(get_metrics_report(y_val_gold, y_val_pred))
print("Classification report (test set)")
print(get_metrics_report(y_test_gold, y_test_pred))
