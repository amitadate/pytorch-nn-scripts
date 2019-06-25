import torch
from torch import autograd, nn
import torch.nn.functional as F

if torch.cuda.is_available():
	print('GPU Available')
else:
	print('CPU Available')



batch_size = 5
input_size = 5
hidden_size = 10
num_classes = 4
learning_rate = 0.001
epochs = 1000

torch.manual_seed(123)

input = autograd.Variable(torch.rand(batch_size, input_size))
target = autograd.Variable(torch.rand(batch_size)* num_classes).long()

# print('input', input)


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


for epoch in range(epochs):
	print('target: ' , target)



	out = M1(input)
	# print('model', M1)
	# print('out',out)

	a,b = out.max(1)

	print('prediction: ',b)

	loss = F.nll_loss(out,target)

	print('loss: ',loss.item())


	M1.zero_grad()
	loss.backward()
	opt.step()
