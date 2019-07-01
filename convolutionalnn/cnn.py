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
from torch.utils.data.sampler import SubsetRandomSampler
# choose one from the below imports
from tqdm import tqdm # scripts
# from tqdm import tqdm_notebook as tqdm # notebooks

# argument parser

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("epochs" , help="display a square of a given number",
                    type=int)
args = parser.parse_args()

# gpu / cpu check


if torch.cuda.is_available():
	print('resource: GPU Available')
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



# torch version and all hyper parameters and path
print('pytorch version: ',torch.__version__)
path = os.getcwd()
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = args.epochs
batch_size = 100
learning_rate = 0.001
time_delay_tqdm = 0.00


# fetching the training and testing datasets from torchvision

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)

test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#


#Training
n_training_samples = 20000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

#Validation
n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

#Test
n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))


# the nn class defination
class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(Net, self).__init__()
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)

        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)


	def forward(self, x):
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 16 *16)

        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)

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





# training loop
start = datetime.datetime.now()

total = 0
correct = 0


pbar1 = tqdm(total = num_epochs,desc = 'training')
for epoch in range (0,num_epochs):


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

	pbar1.update(1)
	time.sleep(time_delay_tqdm)
		# if ((i+1) % 100 == 0):
		# 	print("#")
		# 	print(loss.item())
		# 	print('#')
		#
		# 	print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
pbar1.close()


net_time = datetime.datetime.now()-start

print(' ')
print("total time for training : ",net_time)
print(' ')
print('training loss : ',train_loss_list.pop())
print(' ')




#plot_loss(train_loss_list,num_epochs) # plotting loss vs epochs

 # ----------- TRAINING ENDS -------------- #


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




torch.save(M1.state_dict(), path + '/ckpt/model.ckpt')
print('model params stored: ', path + '/ckpt/model.ckpt')



print(' ')
print('**')
print('Script End')
print('**')
print(' ')





## end of script
