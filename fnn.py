import torch
from torch import autograd, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#from tensorboardX import SummaryWriter
#writer = SummaryWriter()
import datetime

if torch.cuda.is_available():
	print('GPU Available')

else:
	print('CPU Available')




print(torch.__version__)
batch_size = 5
input_size = 5
hidden_size = 10
num_classes = 4
learning_rate = 0.001
epochs = 10

torch.manual_seed(123)

input = autograd.Variable(torch.rand(batch_size, input_size))
target = autograd.Variable(torch.rand(batch_size)* num_classes).long()

# print('input', input)


def plot_loss(loss,epochs):


	idx = []
	for i in range(epochs):
		idx.append(i)


	# x axis values

	x = idx

	# corresponding y axis values
	y = epochs

	# plotting the points
	plt.plot(x, y)

	# naming the x axis
	plt.xlabel('EPOCHS')
	# naming the y axis
	plt.ylabel('LOSS')

	# giving a title to my graph
	plt.title('Generator Loss f-MNIST')
	plt.savefig('Generator Loss f-MNIST.png')

	# function to show the plot
	plt.show()


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


M1 = Net(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
opt = torch.optim.Adam(params = M1.parameters(),lr = learning_rate)


start = datetime.datetime.now()
c = [] # list for tracking time
d = [] # list for tracking loss

for epoch in range(epochs):
	if d != -1:

		print (" ")

		print(' .. ')

		print('epoch: ',epoch)

		print('target: ' , target)
		temp_time = datetime.datetime.now()



		out = M1(input)
		# print('model', M1)
		# print('out',out)

		a,b = out.max(1)

		print('prediction: ',b)

		loss = F.nll_loss(out,target)

		print('loss: ',loss.item())

		d.append(loss.item())


		M1.zero_grad()
		loss.backward()
		opt.step()

		print("time for this epoch: ",datetime.datetime.now()-temp_time)
		print("total time invested in training: ", datetime.datetime.now() -  start)


		print (" ")

		print(' .. ')




plot_loss(d,epochs)





	# print("current time: ", time.time())
