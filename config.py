import os
import torch

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

EMB_PATH = os.path.join(BASE_PATH, "embeddings")

DATA_PATH = os.path.join(BASE_PATH, "datasets")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

