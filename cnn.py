## start of script

print(' ')
print('**')
print('Script Start')
print('**')
print(' ')

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

# gpu / cpu check


if torch.cuda.is_available():
	print('resource: GPU Available')
	device = torch.device("cuda")
	print(' ')
	cmd = 'nvidia-smi'
	os.system(cmd)
	print(' ')
    # os.system(nvidia-smi)

else:
	print('resource: ',cpuinfo.get_cpu_info()['brand'])
	device = torch.device("cpu")
    # os.system()



# torch version and all hyper parameters and path
print('pytorch version: ',torch.__version__)
path = os.getcwd()
input_size = 784
hidden_size = 1
num_classes = 10
num_epochs = 1
batch_size = 100
learning_rate = 0.001
time_delay_tqdm = 0.1


# fetching the training and testing datasets from torchvision
train_dataset = torchvision.datasets.MNIST(root=path + '/data/mnist/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)


test_dataset = torchvision.datasets.MNIST(root=path + '/data/mnist/',
                                          train=False,
                                          transform=transforms.ToTensor(),
										  download=True)


# convert the dataset into input tensors with one of the dimensions being the batch size
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


print('****************')
print(len(test_loader.dataset))
# the nn class defination
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

# the nn object creation
M1 = Net(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)

print(' ')
summary(M1,input_size=(batch_size,input_size))
print(' ')

# criterion defination
criterion = nn.CrossEntropyLoss()
# the opimizer defination
opt = torch.optim.Adam(params = M1.parameters(),lr = learning_rate)

# ----------- TRAINING BEGINS -------------- #



train_loss_list = [] # list for tracking loss

total_step = len(train_loader)

print(' ')
print('training started')
print(' ')

# training loop
start = datetime.datetime.now()

total = 0
correct = 0

for epoch in range(0,num_epochs):
	for i, (images, labels) in enumerate(train_loader):

		temp_time = datetime.datetime.now()

		# get inputs from train_loader as tensors
		images = images.reshape(-1, 28*28).to(device)
		labels = labels.to(device)

		# Forward pass
		outputs = M1(images)
		loss = criterion(outputs, labels)

		_, predicted = torch.max(outputs.data, 1)
		# print(predicted)

		# Add loss to list
		train_loss_list.append(loss.item())

		# Backward and optimize
		M1.zero_grad()
		loss.backward()
		opt.step()

		if (i+1) % 100 == 0:

			#print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
			train_loss_list.append(loss.item())


net_time = datetime.datetime.now()-start
print(' ')
print('training ended')
print(' ')
print("total time for training : ",net_time)
print(' ')
print('train loss: ',train_loss_list.pop())
print(' ')



#plot_loss(train_loss_list,num_epochs) # plotting loss vs epochs

 # ----------- TRAINING ENDS -------------- #

 # ----------- TESTING BEGINS -------------- #

# testing loop

test_runs = len(test_loader)

pbar = tqdm(total = len(test_loader))

print(' ')
print('testing started')
print(' ')


with torch.no_grad():
	total = 0
	correct = 0
	for images,labels in test_loader:
		images = images.reshape(-1,28*28).to(device)
		labels = labels.to(device)
		output = M1(images)
		_, predicted = torch.max(output.data, 1)
		total += labels.size(0)
		correct += (predicted==labels).sum().item()
		pbar.update(1)
		time.sleep(time_delay_tqdm)


pbar.close()

print(' ')
print('testing ended')
print(' ')
print('Number of test runs: ',total)
print(' ')
print('testing accuracy : {} %'.format(100 * correct / total))




# ----------- TESTING ENDS -------------- #


# method to plot graph, input is e - list of values and epochs : number of epochs
def plot_loss(e,epochs):
	idx = []
	for i in range(1,epochs+1):
		idx.append(i)

	j = [x * -1 for x in  e]

	y = j
	x = idx

	# plotting the points
	plt.plot(x, y)

	# naming the x axis
	plt.xlabel('EPOCHS')
	# naming the y axis
	plt.ylabel('Train Accuracy')

	# giving a title to my graph
	plt.title('Training ')
	plt.savefig('train.png')
	print('training plot: train.png')

	# function to show the plot
	#plt.show()
	plt.close()

	print("train accuracy: ", y.pop())








print(' ')
print('**')
print('Script End')
print('**')
print(' ')





## end of script
