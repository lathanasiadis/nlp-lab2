from torch.utils.data import Dataset
from tqdm import tqdm


class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
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

        self.data = X
        self.labels = y
        self.word2idx = word2idx

        # EX2

        self.max_len = 48

        self.sentences = []
        self.lengths = []

        for sentence in X:
            sentence = [self.word2idx.get(word, "<unk>") for word in sentence.split(" ")]
            self.lengths.append(len(sentence))
            if len(sentence) > self.max_len:
                sentence = sentence[:self.max_len]
            else:
                for i in range(self.max_len - len(sentence)):
                    sentence.append(0)
            self.sentences.append(sentence)

    
        #for i in range(100):
        #    print(self.sentences[i])
        #    print(self.lengths[i])
       
        #import matplotlib.pyplot as plt
        #print(min(len(tokenized) for tokenized in tokenized_data))
        #print(max(len(tokenized) for tokenized in tokenized_data))
        #plt.hist([len(tokenized) for tokenized in tokenized_data], bins=30)
        #plt.show()
        
        #raise NotImplementedError        
    
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
        example = self.sentences[index]
        label = self.labels[index]
        length = self.lengths[index]

        return example, label, length
        raise NotImplementedError

