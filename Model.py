# File to return the Deep VO model.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from SE3Comp import *


# DeepVO model
class DeepVO(nn.Module):

	def __init__(self, imageWidth, imageHeight,seqLen,batchsize, activation = 'relu', parameterization = 'default', batchnorm = False, \
		dropout = 0.0, flownet_weights_path = None, numLSTMCells = 1):

		super(DeepVO, self).__init__()


		self.imageWidth = int(imageWidth)
		self.imageHeight = int(imageHeight)
		self.seqLen = int(seqLen)
		self.batchsize = int(batchsize)

		if self.imageWidth < 64 or self.imageHeight < 64:
			raise ValueError('The width and height for an input image must be at least 64 px.')

		# Compute the size of the LSTM input feature vector.
		# There are 6 conv stages (some stages have >1 conv layers), which effectively reduce an 
		# image to 1/64 th of its initial dimensions. Further, the final conv layer has 1024
		# filters, hence, numConcatFeatures = 1024 * (wd/64) * (ht/64) = (wd * ht) / 4
		self.numConcatFeatures = int((self.imageWidth * self.imageHeight) / 4)

		self.activation = activation
		self.parameterization = parameterization
		if parameterization == 'quaternion':
			self.rotationDims = 4
		else:
			self.rotationDims = 3
		self.translationDims = 3

		self.batchnorm = batchnorm
		if dropout <= 0.0:
			self.dropout = False
		else:
			self.dropout = True
			self.drop_ratio = dropout

		self.numLSTMCells = numLSTMCells
	
		
		# Path to FlowNet weights
		if flownet_weights_path is None:
			self.use_flownet = False
		else:
			
			self.use_flownet = True
			self.flownet_weights_path = flownet_weights_path

			


		"""
		Initialize variables required for the network
		"""

		# If we're using batchnorm, do not use bias for the conv layers
		self.bias = not self.batchnorm

		self.conv1   = nn.Conv2d(6, 64, 7, 2, 3, bias = self.bias)
		self.conv2   = nn.Conv2d(64, 128, 5, 2, 2, bias = self.bias)
		self.conv3   = nn.Conv2d(128, 256, 5, 2, 2, bias = self.bias)
		self.conv3_1 = nn.Conv2d(256, 256, 3, 1, 1, bias = self.bias)
		self.conv4   = nn.Conv2d(256, 512, 3, 2, 1, bias = self.bias)
		self.conv4_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = self.bias)
		self.conv5   = nn.Conv2d(512, 512, 3, 2, 1, bias = self.bias)
		self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1, bias = self.bias)
		self.conv6   = nn.Conv2d(512, 1024, 3, 2, 1, bias = self.bias)
		
		#LSTM for rotation and translations
		self.LSTM_R = nn.LSTM(self.numConcatFeatures,1024, self.numLSTMCells)
		self.LSTM_T = nn.LSTM(self.numConcatFeatures,1024, self.numLSTMCells)

		self.h_R = torch.zeros(self.numLSTMCells,self.batchsize,1024)
		self.c_R = torch.zeros(self.numLSTMCells,self.batchsize,1024)
				
		self.h_T = torch.zeros(self.numLSTMCells,self.batchsize,1024)
		self.c_T = torch.zeros(self.numLSTMCells,self.batchsize,1024)


		
		
		self.fc1_R = nn.Linear(1024, 128)
		self.fc1_T = nn.Linear(1024, 128)

		self.fc2_R = nn.Linear(128, 32)
		self.fc2_T = nn.Linear(128, 32)

		if self.parameterization == 'quaternion':
			self.fc_rot = nn.Linear(32, 4)
		else:
			self.fc_rot = nn.Linear(32, 3)
		
		self.fc_trans = nn.Linear(32,3)

		self.se3Layer = SE3Comp()

	def forward(self, x):

		x = (F.leaky_relu(self.conv1(x)))
		x = (F.leaky_relu(self.conv2(x)))
		x = (F.leaky_relu(self.conv3(x)))
		x = (F.leaky_relu(self.conv3_1(x)))
		x = (F.leaky_relu(self.conv4(x)))
		x = (F.leaky_relu(self.conv4_1(x)))
		x = (F.leaky_relu(self.conv5(x)))
		x = (F.leaky_relu(self.conv5_1(x)))
		
		x = ((self.conv6(x))) # No relu at the last conv
		tmp = x.clone()
		
			# Stacking the output from the final conv layer
		x = x.view(self.seqLen,self.numConcatFeatures)
		x=x.unsqueeze(1)


		#print(x.shape)

		o_R, (self.h_R, self.c_R) = self.LSTM_R(x,(self.h_R,self.c_R))
		o_T, (self.h_T, self.c_T) = self.LSTM_T(x,(self.h_T,self.c_T))
			
				
		# Forward pass through the FC layers
		if self.activation == 'relu':
			output_fc1_R = F.relu(self.fc1_R(o_R))
			output_fc1_T = F.relu(self.fc1_T(o_T))
			
			if self.dropout is True:
				output_fc2_R = F.dropout(self.fc2_R(output_fc1_R), p = self.drop_ratio,training = self.training)
				output_fc2_T = F.dropout(self.fc2_T(output_fc1_T), p = self.drop_ratio,training = self.training)
			else:
				output_fc2_R = self.fc2_R(output_fc1_R)
				output_fc2_T = self.fc2_T(output_fc1_T)
		
		elif self.activation == 'selu':
				
			output_fc1_R = F.selu(self.fc1_R(o_R))
			output_fc1_T = F.selu(self.fc1_T(o_T))
			if self.dropout is True:
				output_fc2_R = F.dropout(self.fc2_R(output_fc1_R), p = self.drop_ratio,training = self.training)
				output_fc2_T = F.dropout(self.fc2_T(output_fc1_T), p = self.drop_ratio,training = self.training)
			else:
				output_fc2_R = self.fc2_R(output_fc1_R)
				output_fc2_T = self.fc2_T(output_fc1_T)

		output_rot = self.fc_rot(output_fc2_R)
		output_trans = self.fc_trans(output_fc2_T)
		
		# Forward pass throught the SE3 layer
		#output_se3 = self.se3Layer(Tg, torch.cat((output_trans.view(3,-1),output_rot.view(3,-1)),0).unsqueeze(0)).squeeze(-1) 
		#t = output_se3[:,:3]
		#q = output_se3[:,3:]
		#q = F.normalize(q)

		return output_rot, output_trans, tmp
		#return q,t


	# Initialize the weights of the network
	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				#print('# Linear')
				nn.init.xavier_normal_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.Conv2d):
				#print('$ Conv2d')
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.LSTM):
				#print('% LSTMC')
				for name, param in m.named_parameters():
					if 'weight' in name:
						nn.init.xavier_normal_(param)
						#nn.init.orthogonal(param)
					elif 'bias' in name:
						# Forget gate bias trick: Initially during training, it is often helpful
						# to initialize the forget gate bias to a large value, to help information
						# flow over longer time steps.
						# In a PyTorch LSTM, the biases are stored in the following order:
						# [ b_ig | b_fg | b_gg | b_og ]
						# where, b_ig is the bias for the input gate, 
						# b_fg is the bias for the forget gate, 
						# b_gg (see LSTM docs, Variables section), 
						# b_og is the bias for the output gate.
						# So, we compute the location of the forget gate bias terms as the 
						# middle one-fourth of the bias vector, and initialize them.
						
						# First initialize all biases to zero
						# nn.init.uniform_(param)
						nn.init.constant_(param, 0.)
						bias = getattr(m, name)
						n = bias.size(0)
						start, end = n // 4, n // 2
						bias.data[start:end].fill_(1.)

		# Special weight_init for rotation FCs
		self.fc_rot.weight.data = self.fc_rot.weight.data / 1000.
		#self.fc_rot.weight.data = self.fc_rot.weight.data /100.	 
		# self.fc_trans.weight.data = self.fc_trans.weight.data * 100.
		#self.usePretrainedCNN = True
		#self.load_flownet_weights()
		#if self.use_flownet is True:
		#	self.load_flownet_weights()
		#else:

			


	# Detach LSTM hidden state (i.e., output) and cellstate variables to free up the
	# computation graph. Gradients will NOT flow backward through the timestep where a
	# detach is performed.

	def reset_LSTM_hidden(self):	
		self.h_R = torch.zeros(self.numLSTMCells,self.batchsize,1024)
		self.c_R = torch.zeros(self.numLSTMCells,self.batchsize,1024)
				
		self.h_T = torch.zeros(self.numLSTMCells,self.batchsize,1024)
		self.c_T = torch.zeros(self.numLSTMCells,self.batchsize,1024)

	def detach_LSTM_hidden(self):
		self.h_R = self.h_R.detach()
		self.c_R = self.c_R.detach()
		
		self.h_T = self.h_T.detach()
		self.c_T = self.c_T.detach()

	def load_flownet_weights(self):
		
		if self.use_flownet is True:

			flownet = torch.load(self.flownet_weights_path)
			weights = flownet['state_dict']
		
			if self.batchnorm is False:

				print("Loading weights from flownet (without batchnorm)")

				self.conv1.weight.data = weights["conv1.0.weight"]
				self.conv1.bias.data = weights["conv1.0.bias"]
				

				self.conv2.weight.data = weights["conv2.0.weight"]
				self.conv2.bias.data = weights["conv2.0.bias"]
				
				self.conv3.weight.data = weights["conv3.0.weight"]
				self.conv3.bias.data = weights["conv3.0.bias"]
				
				self.conv3_1.weight.data = weights["conv3_1.0.weight"]
				self.conv3_1.bias.data = weights["conv3_1.0.bias"]
				
				self.conv4.weight.data = weights["conv4.0.weight"]
				self.conv4.bias.data = weights["conv4.0.bias"]
				
				self.conv4_1.weight.data = weights["conv4_1.0.weight"]
				self.conv4_1.bias.data = weights["conv4_1.0.bias"]
				
				self.conv5.weight.data = weights["conv5.0.weight"]
				self.conv5.bias.data = weights["conv5.0.bias"]
				
				self.conv5_1.weight.data = weights["conv5_1.0.weight"]
				self.conv5_1.bias.data = weights["conv5_1.0.bias"]
				
				self.conv6.weight.data = weights["conv6.0.weight"]
				self.conv6.bias.data = weights["conv6.0.bias"]
				
			else:
				


				self.conv1.weight.data = weights["conv1.0.weight"]
				self.conv1_bn.weight.data = weights["conv1.1.weight"]
				self.conv1_bn.bias.data = weights["conv1.1.bias"]
				self.conv1_bn.running_mean.data = weights["conv1.1.running_mean"]
				self.conv1_bn.running_var.data = weights["conv1.1.running_var"]

				self.conv2.weight.data = weights["conv2.0.weight"]
				self.conv2_bn.weight.data = weights["conv2.1.weight"]
				self.conv2_bn.bias.data = weights["conv2.1.bias"]
				self.conv2_bn.running_mean.data = weights["conv2.1.running_mean"]
				self.conv2_bn.running_var.data = weights["conv2.1.running_var"]

				self.conv3.weight.data = weights["conv3.0.weight"]
				self.conv3_bn.weight.data = weights["conv3.1.weight"]
				self.conv3_bn.bias.data = weights["conv3.1.bias"]
				self.conv3_bn.running_mean.data = weights["conv3.1.running_mean"]
				self.conv3_bn.running_var.data = weights["conv3.1.running_var"]

				self.conv3_1.weight.data = weights["conv3_1.0.weight"]
				self.conv3_1_bn.weight.data = weights["conv3_1.1.weight"]
				self.conv3_1_bn.bias.data = weights["conv3_1.1.bias"]
				self.conv3_1_bn.running_mean.data = weights["conv3_1.1.running_mean"]
				self.conv3_1_bn.running_var.data = weights["conv3_1.1.running_var"]

				self.conv4.weight.data = weights["conv4.0.weight"]
				self.conv4_bn.weight.data = weights["conv4.1.weight"]
				self.conv4_bn.bias.data = weights["conv4.1.bias"]
				self.conv4_bn.running_mean.data = weights["conv4.1.running_mean"]
				self.conv4_bn.running_var.data = weights["conv4.1.running_var"]

				self.conv4_1.weight.data = weights["conv4_1.0.weight"]
				self.conv4_1_bn.weight.data = weights["conv4_1.1.weight"]
				self.conv4_1_bn.bias.data = weights["conv4_1.1.bias"]
				self.conv4_1_bn.running_mean.data = weights["conv4_1.1.running_mean"]
				self.conv4_1_bn.running_var.data = weights["conv4_1.1.running_var"]

				self.conv5.weight.data = weights["conv5.0.weight"]
				self.conv5_bn.weight.data = weights["conv5.1.weight"]
				self.conv5_bn.bias.data = weights["conv5.1.bias"]
				self.conv5_bn.running_mean.data = weights["conv5.1.running_mean"]
				self.conv5_bn.running_var.data = weights["conv5.1.running_var"]

				self.conv5_1.weight.data = weights["conv5_1.0.weight"]
				self.conv5_1_bn.weight.data = weights["conv5_1.1.weight"]
				self.conv5_1_bn.bias.data = weights["conv5_1.1.bias"]
				self.conv5_1_bn.running_mean.data = weights["conv5_1.1.running_mean"]
				self.conv5_1_bn.running_var.data = weights["conv5_1.1.running_var"]

				self.conv6.weight.data = weights["conv6.0.weight"]
				self.conv6_bn.weight.data = weights["conv6.1.weight"]
				self.conv6_bn.bias.data = weights["conv6.1.bias"]
				self.conv6_bn.running_mean.data = weights["conv6.1.running_mean"]
				self.conv6_bn.running_var.data = weights["conv6.1.running_var"]

		else:
			print("Loading from pretrained CNN")

			weights = torch.load('/u/sharmasa/Documents/DeepVO_CNN/cache/KITTI/seq1/models/best.pt')
			
			self.conv1.weight.data = weights.conv1.weight.data
			self.conv1.bias.data = weights.conv1.bias.data
			
			self.conv2.weight.data = weights.conv2.weight.data
			self.conv2.bias.data = weights.conv2.bias.data
			
			self.conv3.weight.data = weights.conv3.weight.data
			self.conv3.bias.data = weights.conv3.bias.data

			self.conv3_1.weight.data = weights.conv3_1.weight.data
			self.conv3_1.bias.data = weights.conv3_1.bias.data
			
			self.conv4.weight.data = weights.conv4.weight.data
			self.conv4.bias.data = weights.conv4.bias.data
			
			self.conv4_1.weight.data = weights.conv4_1.weight.data
			self.conv4_1.bias.data = weights.conv4_1.bias.data
			
			self.conv5.weight.data = weights.conv5.weight.data
			self.conv5.bias.data = weights.conv5.bias.data
			
			
			self.conv5_1.weight.data = weights.conv5_1.weight.data
			self.conv5_1.bias.data = weights.conv5_1.bias.data
			
			self.conv6.weight.data = weights.conv6.weight.data
			self.conv6.bias.data = weights.conv6.bias.data

			self.fc1_R.weight.data = weights.fc1_R.weight.data
			self.fc1_T.weight.data = weights.fc1_T.weight.data
			self.fc1_R.bias.data = weights.fc1_R.bias.data
			self.fc1_T.bias.data = weights.fc1_T.bias.data

			self.fc2_R.weight.data = weights.fc2_R.weight.data
			self.fc2_T.weight.data = weights.fc2_T.weight.data
			self.fc2_R.bias.data = weights.fc2_R.bias.data
			self.fc2_T.bias.data = weights.fc2_T.bias.data


			self.fc_rot.weight.data = weights.fc_rot.weight.data
			self.fc_trans.weight.data = weights.fc_trans.weight.data
			self.fc_rot.bias.data = weights.fc_rot.bias.data
			self.fc_trans.bias.data = weights.fc_trans.bias.data









		
