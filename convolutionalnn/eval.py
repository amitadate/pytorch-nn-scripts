# imports
import numpy as np
import torch
from torch import autograd, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import tqdm
import cpuinfo
import os
import time
import argparse
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
# choose one from the below imports
from tqdm import tqdm # scripts
# from tqdm import tqdm_notebook as tqdm # notebooks

# argument parser


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("mode" , help="display a square of a given number")
args = parser.parse_args()

if torch.cuda.is_available():
	print('resource: GPU Available \n')
	device = torch.device("cuda")
	print(' ')
	cmd = 'gpustat'
	os.system(cmd)
	print(' ')
    # os.system(nvidia-smi)

else:
	print('resource: ',cpuinfo.get_cpu_info()['brand'])
	device = torch.device("cpu")
    # os.system()


print('pytorch version: ',torch.__version__)
path = os.getcwd()
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001
time_delay_tqdm = 0.00
eval_mode = args.mode


class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super().__init__()
		self.h1 = nn.Linear(input_size, hidden_size)
		self.h2 = nn.Linear(hidden_size, num_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.h1(x)
		x = self.relu(x)
		x = self.h2(x)
		# x = F.softmax(x, dim = 1)
		return x



import torch
ckptl = torch.load(path + '/ckpt/model.ckpt')
Net1 = Net(input_size, hidden_size, num_classes).to(device)
Net1.load_state_dict(ckptl)
print(' ')
print('model loaded: ', path + '/ckpt/model.ckpt')


test_dataset = torchvision.datasets.MNIST(root=path + '/data/mnist/',
                                          train=False,
                                          transform=transforms.ToTensor(),
										  download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

train_dataset = torchvision.datasets.MNIST(root=path + '/data/mnist/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


# testing loop
start = datetime.datetime.now()

if eval_mode == 'train':
	bandwidth = train_loader
	runs = len(train_loader)

if eval_mode == 'test':
	bandwidth = test_loader
	runs = len(test_loader)
print(' ')
print("mode for evaluation : ",eval_mode)
print(' ')

with torch.no_grad():
	pbar2 = tqdm(total=runs, desc = 'evaluating loaded model')
	total = 0
	correct = 0
	for images,labels in bandwidth:
		temp_time = datetime.datetime.now()
		images = images.reshape(-1,28*28).to(device)
		labels = labels.to(device)
		output = Net1(images)
		_, predicted = torch.max(output.data, 1)
		total += labels.size(0)
		correct += (predicted==labels).sum().item()
		pbar2.update(1)
		time.sleep(time_delay_tqdm)


pbar2.close()

net_time = datetime.datetime.now()-start

print(' ')
print("total time for eval : ",net_time)
print(' ')
print('Number of ' + eval_mode+ ' samples: ',total)
print(' ')
print(str(eval_mode) +  ' accuracy : {} %'.format(100 * correct / total))
print(' ')
