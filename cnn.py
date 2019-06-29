# imports
import torch
from torch import autograd, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime


# gpu / cpu check
if torch.cuda.is_available():
	print('GPU Available')

else:
	print('CPU Available')


# torch version and all hyper parameters
print(torch.__version__)
batch_size = 5
input_size = 5
hidden_size = 10
num_classes = 4
learning_rate = 0.001
epochs = 10000

# testing
from tqdm import tqdm
import time

for i in tqdm(range(100)):

    # time.sleep(0.1)
    pass
