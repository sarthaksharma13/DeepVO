from lieFunctions import *
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob


import warnings
warnings.filterwarnings("ignore")

from lieFunctions import euler_to_rotMat
	

def getGroundTruthTrajectory(seq, seqLength, dataDir):
	
	cameraTraj = np.empty([seqLength,3])
	fullT = np.empty([seqLength,4,4]);
	# poses = open(os.path.join(dataDir, 'poses', str(seq).zfill(2) + '.txt'))
	poses = np.loadtxt(os.path.join(dataDir, 'poses', str(seq).zfill(2) + '.txt'))
	# poses = open("/data/milatmp1/sharmasa/"+ dataset + "/dataset/poses/" + str(seq).zfill(2) + ".txt").readlines()
	for frame in range(seqLength):
		# pose = np.concatenate((np.asarray([(float(i)) for i in poses[frame].split(' ')]).reshape(3,4) , [[0.0,0.0,0.0,1.0]]), axis=0);
		pose = np.concatenate((np.asarray(poses[frame]).reshape(3, 4), [[0., 0., 0., 1.]]), axis = 0)
		cameraTraj[frame,:] = pose[0:3,3].T;
		fullT[frame,:] = pose

	return cameraTraj,fullT;


def plotSequence(expDir, seq, seqLength, trajectory, dataDir, cmd, epoch):

	T = np.eye(4);
	estimatedCameraTraj = np.empty([seqLength,3])
	gtCameraTraj,fullT = getGroundTruthTrajectory(seq, seqLength, dataDir);

	# Extract the camera centres from all the frames

	# First frame as the world origin
	estimatedCameraTraj[0] = np.zeros([1,3]);
	for frame in range(seqLength-1):

		# Output is pose of frame i+1 with respect to frame i
		relativePose = trajectory[frame,:]

		if cmd.outputParameterization == 'se3':
			estimatedCameraTraj[frame+1] = relativePose[:3]
		else:
			

			if cmd.outputParameterization == 'default':
				R = axisAngle_to_rotMat(np.transpose(relativePose[:3]))
				t = np.reshape(relativePose[3:],(3,1))
			elif cmd.outputParameterization == 'euler':
				R = euler_to_rotMat(relativePose[0]/10.,relativePose[1]/10.,relativePose[2]/10., seq='xyz')
				t = np.reshape(relativePose[3:],(3,1))
			elif cmd.outputParameterization == 'quaternion' :
				R = quat_to_rotMat(np.transpose(relativePose[:4]))
				t = np.reshape(relativePose[4:],(3,1))

			T_1 = fullT[frame,:]
			T_2 = fullT	[frame+1,:]
			T_r_gt = np.dot(np.linalg.inv(T_1),T_2)
			R_gt = T_r_gt[0:3,0:3]
			t_gt = T_r_gt[0:3,3].reshape(3,-1)
			
			T_r = np.concatenate( ( np.concatenate([R,t],axis=1) , [[0.0,0.0,0.0,1.0]] ) , axis = 0 )



			
			
			# With respect to the first frame
			T_abs = np.dot(T,T_r);
			# Update the T matrix till now.
			T = T_abs

			# Get the origin of the frame (i+1), ie the camera center
			estimatedCameraTraj[frame+1] = np.transpose(T[0:3,3])

		

	# Get the ground truth camera trajectory
	

	# Plot the estimated and groundtruth trajectories
	x_gt = gtCameraTraj[:,0]
	z_gt = gtCameraTraj[:,2]

	x_est = estimatedCameraTraj[:,0]
	z_est = estimatedCameraTraj[:,2]
	
	fig,ax = plt.subplots(1)
	ax.plot(x_gt,z_gt, 'c', label = "ground truth")
	ax.plot(x_est,z_est, 'm', label= "estimated")
	ax.legend()
	fig.savefig(os.path.join(expDir, 'plots', 'traj', str(seq).zfill(2), 'traj_' + str(epoch).zfill(3)))













	



