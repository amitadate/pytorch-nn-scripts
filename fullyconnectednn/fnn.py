## start of script

print(' ')
print('**')
print('Script Start')
print('**')
print(' ')

# imports
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



# torch version and all hyper parameters
print('pytorch version: ',torch.__version__)
batch_size = 10
input_size = 10
hidden_size = 1
num_classes = 5
learning_rate = 0.001
epochs = 5000
time_delay_tqdm = 0.000


# defining input and target
torch.manual_seed(123)

input = autograd.Variable(torch.rand(batch_size, input_size)).to(device)
target = autograd.Variable(torch.rand(batch_size)* num_classes).long().to(device)
#
# print('input', input)
# print('target', target)

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



# the nn class defination
class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super().__init__()
		self.h1 = nn.Linear(input_size, hidden_size)
		self.h2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		x = self.h1(x)
		x = torch.tanh(x)
		x = self.h2(x)
		x = F.softmax(x, dim = 1)
		return x


# the nn object creation
M1 = Net(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)

# the opimizer defination
opt = torch.optim.Adam(params = M1.parameters(),lr = learning_rate)

# training loop
start = datetime.datetime.now()
c = [] # list for tracking time
d = [] # list for tracking loss


# tqdm object creation as progress bar
#
# import tqdm
# pbar = tqdm.tqdm(total=100) #choose number of levels on progress bar
# outer = tqdm.tqdm(total=epochs, position=0)
print(' ')
print('input ', input)
print('input_size ', input_size)
print('input_shape ', input.shape)
summary(M1,input_size=input.shape)
print(' ')
for epoch in tqdm(range(0,epochs),desc = 'training'):


# for epoch in range(epochs):
	if d != -1:

		# print (" ")
		#
		# print(' .. ')
		#
		# print('epoch: ',epoch)
		#
		# print('target: ' , target)


		temp_time = datetime.datetime.now()


		out = M1(input) # feed forward through the net
		# print('model', M1)
		# print('out',out)

		a,b = out.max(1) # storing output in b

		# print('prediction: ',b) # displaying the nn output

		loss = F.nll_loss(out,target) # defining the loss

		# print('loss: ',loss.item()) # displaying the loss

		d.append(loss.item()) # storing loss in list for logging purposes


		M1.zero_grad() # making gradients zero

		loss.backward() # backward pass ono autograd variables

		opt.step() # one optmizer step
		#
		# print("time for this epoch: ",datetime.datetime.now()-temp_time)
		#
		# print("total time invested in training: ", datetime.datetime.now() -  start)
		#
		#
		# print ("percentage of training complete: ", ((1-(epoch/epochs)) * 100))

		time.sleep(time_delay_tqdm) #update the tqdm object after the sleeping time
net_time = datetime.datetime.now()-start

print("total time: ",net_time)





plot_loss(d,epochs) # plotting loss vs epochs












print(' ')
print('**')
print('Script End')
print('**')
print(' ')





## end of script
