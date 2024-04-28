import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from nltk.tokenize import word_tokenize, TweetTokenizer

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx, tweets=False, verbose=False):

        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """
        # EX2
        tt = TweetTokenizer()
        tokenizer = tt.tokenize if tweets else word_tokenize
        self.data = list(map(tokenizer, X))
        
        #  90% quantile of sentence length (in number of tokens)
        lens = list(map(len, self.data))
        lens.sort()
        self.max_len = lens[int(0.9 * len(lens))]

        self.labels = y
        self.word2idx = word2idx

        if verbose:
            for i in range(10):
                print("Sentence: {}\nLabel: {}".format(X[i], y[i]))
        
            print("max_len = {}".format(self.max_len))
            print("<unk> embedding: {}".format(self.word2idx["<unk>"]))
            print("==============================")

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3
        item = self.data[index][:self.max_len]
        item_len = min(len(item), self.max_len)
        toks = list(map(
            lambda x: self.word2idx.get(x) or self.word2idx["<unk>"], item
        ))
        return np.pad(toks, (0, self.max_len - item_len)), self.labels[index], item_len
