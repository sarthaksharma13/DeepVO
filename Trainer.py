"""
Trainer class. Handles training and validation
"""
from helpers import get_gpu_memory_map
from KITTIDataset import KITTIDataset
from Model import DeepVO
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from random import randint

class Trainer():

	def __init__(self, args, epoch, model, train_set, val_set, loss_fn, optimizer, scheduler = None, \
		gradClip = None):

		self.args = args
		self.maxEpochs = self.args.nepochs
		self.curEpoch = epoch

		# Model to train
		self.model = model

		# Train and validataion sets (Dataset objects)
		self.train_set = train_set
		self.val_set = val_set

		# Loss function
		self.loss_fn = nn.MSELoss(reduction = 'sum')

		# Variables to hold loss
		self.loss = torch.zeros(1, dtype = torch.float32).cuda()
		self.loss_R = torch.zeros(1, dtype = torch.float32).cuda()
		self.loss_T = torch.zeros(1, dtype = torch.float32).cuda()
		
		# Optimizer
		self.optimizer = optimizer

		# Scheduler
		self.scheduler = scheduler


	# Train for one epoch
	def train(self):

		# Switch model to train mode
		self.model.train()

		# Check if maxEpochs have elapsed
		if self.curEpoch >= self.maxEpochs:
			print('Max epochs elapsed! Returning ...')
			return

		# Variables to store stats
		rotLosses = []
		transLosses = []
		totalLosses = []
		

		# Handle debug mode here
		if self.args.debug is True:
			numTrainIters = self.args.debugIters
		else:
			numTrainIters = len(self.train_set)

		elapsedFrames = 0

		# Choose a generator (for iterating over the dataset, based on whether or not the 
		# sbatch flag is set to True). If sbatch is True, we're probably running on a cluster
		# and do not want an interactive output. So, could suppress tqdm and print statements
		if self.args.sbatch is True:
			gen = range(numTrainIters)
		else:
			gen = trange(numTrainIters)

		# Store input and label tensors
		
		inputTensor = None
		labelTensor_trans = None
		labelTensor_rot = None

		# Run a pass of the dataset
		for i in gen:

			if self.args.profileGPUUsage is True:
				gpu_memory_map = get_gpu_memory_map()
				tqdm.write('GPU usage: ' + str(gpu_memory_map[0]), file = sys.stdout)

			# Get the next frame
			inp, rot_gt, trans_gt, seq, frame1, frame2, endOfSeq = self.train_set[i]
			if inputTensor is None:
				inputTensor = inp.clone()
				labelTensor_rot = rot_gt.unsqueeze(0).clone()
				labelTensor_trans = trans_gt.unsqueeze(0).clone()
			else:
				inputTensor = torch.cat((inputTensor,inp.clone()),0)
				labelTensor_rot = torch.cat((labelTensor_rot,rot_gt.unsqueeze(0).clone()),0)
				labelTensor_trans = torch.cat((labelTensor_trans, trans_gt.unsqueeze(0).clone()),0)


			# Handle debug mode here. Force execute the below if statement in the
			# last debug iteration
			if self.args.debug is True:
				if i == numTrainIters - 1:
					endOfSeq = True

			elapsedFrames += 1

			
			# if endOfSeq is True:
			if elapsedFrames >= self.args.seqLen or endOfSeq is True:

				# Flush gradient buffers for next forward pass
				self.model.zero_grad()

				
				rot_pred, trans_pred, tmp = self.model.forward(inputTensor)

				
				

				self.loss = sum([100*self.loss_fn(rot_pred, labelTensor_rot),self.loss_fn(trans_pred, labelTensor_trans)])
				curloss_rot = Variable(torch.dist(rot_pred, labelTensor_rot) ** 2, requires_grad = False)
				curloss_trans = Variable(torch.dist(trans_pred, labelTensor_trans) ** 2 , requires_grad = False)

				# re initialize
				inputTensor = None
				labelTensor_trans = None
				labelTensor_rot = None
				elapsedFrames = 0

				paramsDict = self.model.state_dict()
				reg_loss_R = None
				reg_loss_R = paramsDict['LSTM_R.weight_ih_l0'].norm(2)
				reg_loss_R += paramsDict['LSTM_R.weight_hh_l0'].norm(2)
				reg_loss_R += paramsDict['LSTM_R.bias_ih_l0'].norm(2)
				reg_loss_R += paramsDict['LSTM_R.bias_hh_l0'].norm(2)
				if self.args.numLSTMCells==2:
					reg_loss_R = paramsDict['LSTM_R.weight_ih_l1'].norm(2)
					reg_loss_R += paramsDict['LSTM_R.weight_hh_l1'].norm(2)
					reg_loss_R += paramsDict['LSTM_R.bias_ih_l1'].norm(2)
					reg_loss_R += paramsDict['LSTM_R.bias_hh_l1'].norm(2)
				
				reg_loss_T = None
				reg_loss_T = paramsDict['LSTM_T.weight_ih_l0'].norm(2)
				reg_loss_T += paramsDict['LSTM_T.weight_hh_l0'].norm(2)
				reg_loss_T += paramsDict['LSTM_T.bias_ih_l0'].norm(2)
				reg_loss_T += paramsDict['LSTM_T.bias_hh_l0'].norm(2)
				if self.args.numLSTMCells==2:
					reg_loss_T = paramsDict['LSTM_T.weight_ih_l1'].norm(2)
					reg_loss_T += paramsDict['LSTM_T.weight_hh_l1'].norm(2)
					reg_loss_T += paramsDict['LSTM_T.bias_ih_l1'].norm(2)
					reg_loss_T += paramsDict['LSTM_T.bias_hh_l1'].norm(2)
					
				totalregLoss = sum([reg_loss_R ,reg_loss_T])
				self.loss = sum([self.args.gamma * totalregLoss, self.loss])
				
				# Compute gradients
				self.loss.backward()

				# Rotation Grad norm 
				paramIt=0;
				rotgradNorm=0
				rotParameters=[]
				for p in self.model.parameters():
					paramIt+=1;
					if paramIt in range(19,27):
						rotParameters.append(p)
						rotgradNorm+=(p.grad.data.norm(2.) ** 2.) 
				rotgradNorm = rotgradNorm ** (1. / 2)
								
				# Translation Grad norm 
				paramIt=0;
				transgradNorm=0
				transParameters=[]
				for p in self.model.parameters():
					paramIt+=1;
					if paramIt in range(27,35):
						
						transParameters.append(p)
						transgradNorm+=(p.grad.data.norm(2.)**2.)
				transgradNorm = transgradNorm ** (1./2)

				tqdm.write('Before clipping, Rotation gradNorm: ' + str(rotgradNorm) + ' Translation gradNorm: ' + str(transgradNorm))

				# Perform gradient clipping, if enabled
				if self.args.gradClip is not None:
					
					torch.nn.utils.clip_grad_norm_(rotParameters, self.args.gradClip)
					torch.nn.utils.clip_grad_norm_(transParameters, self.args.gradClip)
					paramIt=0;
					rotgradNorm=0
					for p in self.model.parameters():
						paramIt+=1;
						if paramIt in range(19,27) :
							rotgradNorm+=(p.grad.data.norm(2.) ** 2.) 
					rotgradNorm = rotgradNorm ** (1. / 2)
								
				
					paramIt=0;
					transgradNorm=0
					for p in self.model.parameters():
						paramIt+=1;
						if paramIt in range(27,35):
							transgradNorm+=(p.grad.data.norm(2.)**2.)
					transgradNorm = transgradNorm ** (1./2)

					tqdm.write('After clipping, Rotation gradNorm: ' + str(rotgradNorm) + ' Translation gradNorm: ' + str(transgradNorm))


				# Update parameters
				self.optimizer.step()
	
				curloss_rot = curloss_rot.detach().cpu().numpy()	
				curloss_trans = curloss_trans.detach().cpu().numpy()
				rotLosses.append(curloss_rot)
				transLosses.append(curloss_trans)
				totalLosses.append(curloss_rot + curloss_trans)

				# Print stats
				tqdm.write('Rot Loss: ' + str(np.mean(rotLosses)) + ' Trans Loss: ' + \
					str(np.mean(transLosses)), file = sys.stdout)
				tqdm.write('Total Loss: ' + str(np.mean(totalLosses)), file = sys.stdout)
				# If it's the end of sequence, reset hidden states
				if endOfSeq is True:
					self.model.reset_LSTM_hidden()
				self.model.detach_LSTM_hidden()
	
				# Reset loss variables
				self.loss = torch.zeros(1, dtype = torch.float32).cuda()
		# Return loss logs for further analysis
		return rotLosses, transLosses, totalLosses


	# Run one epoch of validation
	def validate(self):
		
		self.model.eval()
		# Run a pass of the dataset
		traj_pred = None
		
		# Variables to store stats
		rotLosses = []
		transLosses = []
		totalLosses = []
		
		# Handle debug switch here
		if self.args.debug is True:
			numValIters = self.args.debugIters
		else:
			numValIters = len(self.val_set)

		elapsedFrames=0;
		if self.args.sbatch is True:
			gen = range(numValIters)
		else:
			gen = trange(numValIters)

		inputTensor = None
		labelTensor_rot = None
		labelTensor_trans = None

		for i in gen:

			if self.args.profileGPUUsage is True:
				gpu_memory_map = get_gpu_memory_map()
				tqdm.write('GPU usage: ' + str(gpu_memory_map[0]), file = sys.stdout)

			# Get the next frame
			inp, rot_gt, trans_gt, seq, frame1, frame2, endOfSeq = self.val_set[i]
			if inputTensor is None:
				inputTensor = inp.clone()
				labelTensor_rot = rot_gt.unsqueeze(0).clone()
				labelTensor_trans = trans_gt.unsqueeze(0).clone()
			else:
				inputTensor = torch.cat((inputTensor,inp.clone()),0)
				labelTensor_rot = torch.cat((labelTensor_rot,rot_gt.unsqueeze(0).clone()),0)
				labelTensor_trans = torch.cat((labelTensor_trans, trans_gt.unsqueeze(0).clone()),0)
			
			
			elapsedFrames+=1

			if elapsedFrames>=self.args.seqLen or endOfSeq is True:

				
				# Feed it through the model
				rot_pred, trans_pred,_ = self.model.forward(inputTensor)

				curloss_rot = Variable(torch.dist(rot_pred, labelTensor_rot) ** 2, requires_grad = False)
				curloss_trans = Variable(torch.dist(trans_pred, labelTensor_trans) ** 2 , requires_grad = False)
				
				inputTensor = None
				labelTensor_trans = None
				labelTensor_rot = None
				elapsedFrames=0;
			
				if traj_pred is None:
					traj_pred = np.concatenate((rot_pred.data.cpu().numpy().squeeze(1), \
						trans_pred.data.cpu().numpy().squeeze(1)), axis = 1)
				else:
					
					cur_pred = np.concatenate((rot_pred.data.cpu().numpy().squeeze(1), \
						trans_pred.data.cpu().numpy().squeeze(1)), axis = 1)
					
					traj_pred = np.concatenate((traj_pred, cur_pred), axis = 0)

				rotLosses.append(curloss_rot)
				transLosses.append(curloss_trans)
				totalLosses.append(curloss_rot + curloss_trans)

				# Deattach for the next forward pass
				self.model.detach_LSTM_hidden()

				

				if endOfSeq is True:
					# Print stats
					tqdm.write('Rot Loss: ' + str(np.mean(rotLosses)) + ' Trans Loss: ' + \
						str(np.mean(transLosses)), file = sys.stdout)
					tqdm.write('Total Loss: ' + str(np.mean(totalLosses)), file = sys.stdout)
					# Write predicted trajectory to file
					saveFile = os.path.join(self.args.expDir, 'plots', 'traj', str(seq).zfill(2), \
						'traj_' + str(self.curEpoch).zfill(3) + '.txt')
					np.savetxt(saveFile, traj_pred, newline = '\n')
				
					# Reset variable, to store new trajectory later on
					traj_pred = None
		
					# Reset LSTM hidden states
					self.model.reset_LSTM_hidden()
					# Deattach for the next forward pass
					self.model.detach_LSTM_hidden()

					rotLosses = []
					transLosses = []
					totalLosses = []




		

		# Return loss logs for further analysis
		return rotLosses, transLosses, totalLosses
