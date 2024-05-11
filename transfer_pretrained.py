from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model",
    choices=["twitter_roberta", "roberta_large", "bertweet", "financial"],
    help="""Pretrained model""",
    default="twitter_roberta"
)
parser.add_argument("-d", "--dataset",
    choices=["mr", "semeval"],
    help="""Dataset used to train and evaluate the model.""",
    default="semeval"
)

args = parser.parse_args()


args_to_models = {"twitter_roberta": 'cardiffnlp/twitter-roberta-base-sentiment',
                  "roberta_large": 'siebert/sentiment-roberta-large-english',
                  "bertweet": "finiteautomata/bertweet-base-sentiment-analysis",
                  "financial": "ahmedrachid/FinancialBERT-Sentiment-Analysis"}
                  

PRETRAINED_MODEL = args_to_models[args.model]

DATASET = "MR" if args.dataset == "mr" else "Semeval2017A"


LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'finiteautomata/bertweet-base-sentiment-analysis': {
        'POS': 'positive',
        'NEG': 'negative',
        'NEU': 'neutral',
    },
}

if __name__ == '__main__':
    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(list(le.classes_))

    # define a proper pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=PRETRAINED_MODEL)

    y_pred = []
    for x in tqdm(X_test):
        # TODO: Main-lab-Q6 - get the label using the defined pipeline
        label = sentiment_pipeline(x)[0]['label']
        y_pred.append(LABELS_MAPPING[PRETRAINED_MODEL][label]
                      if PRETRAINED_MODEL in LABELS_MAPPING else label)

    y_pred = le.transform(y_pred)
    print(f'\nDataset: {DATASET}\nPre-Trained model: {PRETRAINED_MODEL}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')
