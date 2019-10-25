import numpy as np
from numpy.random import *
import sys,itertools,os
from scipy.stats import pearsonr
from scipy.stats import chi2
from math import exp
from copy import deepcopy
from metric import *

def clipped(x,a,b):
	if x<a:
		return a
	if x>b:
		return b
	return x

def NTI(answer,VLDB=True):

	n=len(answer)
	m=len(answer[0])
	r=[len(answer[0][j]) for j in range(0,m)]

	truth=[[uniform(0,1) for k in range(0,r[j])] for j in range(0,m)]
	sigma=[uniform(0,1) for i in range(0,n)]
	quality=[1.0/n for i in range(0,n)]

	last_truth=np.array([])

	while True:

		if len(last_truth)!=0:
			maxdiff=0.0
			for j in range(0,m):
				for k in range(0,r[j]):
					maxdiff=max(maxdiff,abs(last_truth[j][k]-truth[j][k]))
			if maxdiff<1E-5:
				break
			else:
				pass

		last_truth=deepcopy(truth)

		for j in range(0,m):
			for k in range(0,r[j]):
				truth[j][k]=0.0
				up=0.0
				down=0.0
				count=0
				for i in range(0,n):
					if answer[i][j][k]!=None:
						up+=answer[i][j][k]*quality[i]
						down+=quality[i]
						count+=1
				if down==0:
					truth[j][k]==None
				else:
					truth[j][k]=up/down

		for i in range(0,n):
			sigma[i]=0.0
			acc_n=0
			for j in range(0,m):
				for k in range(0,r[j]):
					if answer[i][j][k]!=None:
						sigma[i]+=(answer[i][j][k]-truth[j][k])**2
						acc_n+=1
			sigma[i]=np.sqrt(sigma[i]/acc_n)

		if not VLDB:

			sum_q=0.0
			for i in range(0,len(quality)):
				quality[i]=1.0/sigma[i]**2
				sum_q+=quality[i]
			for i in range(0,len(quality)):
				quality[i]=quality[i]/sum_q+1E-10
			
		else:

			sum_quality=0.0
			for i in range(0,n):
				Ns=0
				up=1E-10
				down=1E-10
				for j in range(0,m):
					for k in range(0,r[j]):
						if answer[i][j][k]!=None:
							Ns+=1
							down+=(answer[i][j][k]-truth[j][k])**2
				up+=chi2.ppf(q=0.95,df=Ns)
				quality[i]=up/down+1E-10

			quality_sum=sum(quality)
			for i in range(0,n):
				quality[i]/=quality_sum

	return truth,None,sigma,quality

def FTI(answer,VLDB=False,ground_truth=None):

	n=len(answer)
	m=len(answer[0])
	r=[len(answer[0][j]) for j in range(0,m)]

	#Step 0: Give random initial numbers
	truth=[[uniform(0.4,0.6) for k in range(0,r[j])] for j in range(0,m)]
	bias=[[uniform(-0.1,0.1) for j in range(0,m)] for i in range(0,n)]
	sigma=[0.0 for i in range(0,n)]
	quality=[1.0/n for i in range(0,n)]
	last_truth=np.array([])


	while True:

		TP,TN,FP,FN=confusion_matrix(esti_truth=truth,truth=ground_truth)
		print(1.0*(TP+TN)/(TP+TN+FP+FN),end=' ')

		if len(last_truth)!=0:
			maxdiff=-1
			for j in range(0,m):
				for k in range(0,r[j]):
					if last_truth[j][k]!=None and truth[j][k]!=None: 
						maxdiff=max(maxdiff,abs(last_truth[j][k]-truth[j][k]))
			print(maxdiff,end=' ')
			if maxdiff<1E-5 and maxdiff>=0:
				break
			else:
				pass

		last_truth=deepcopy(truth)

		#Step 0: Fairness correction

		# 
		# version 1: pull everything to 0.5
		# votes=[[0.0,0.0] for i in range(0,m)]
		# for j in range(0,m):
		# 	for k in range(0,len(truth[j])):
		# 		votes[j][0]+=1-truth[j][k]
		# 		votes[j][1]+=truth[j][k]
		# accept_rate=[]
		# for j in range(0,m):
		# 	accept_rate.append(1-abs(votes[j][0]-votes[j][1])/2.0/max(votes[j]))
		# print(accept_rate)
		# 

		#version 2:pull everything to average
		criminal_ratio=[sum(truth[j])/len(truth[j]) for j in range(0,m)]
		criminal_average=0.0
		criminal_count=0.0
		for j in range(0,m):
			for k in range(0,m):
				criminal_average+=truth[j][k]
				criminal_count+=1
		criminal_average/=criminal_count
		for i in range(0,m):
			criminal_ratio[i]=criminal_ratio[i]-criminal_average
		accept_ratio=[]
		for i in range(0,m):
			if criminal_ratio[i]>0:
				accept_ratio.append(1-criminal_ratio[i]/criminal_average)
			else:
				accept_ratio.append(1-abs(criminal_ratio[i])/(1-criminal_average))
		print(accept_ratio)
		#


		#Step 1: Estimate Truth
		for j in range(0,m):
			for k in range(0,r[j]):
				truth[j][k]=0.0
				acc_q=0.0
				for i in range(0,n):
					if answer[i][j][k]!=None:
						# if (votes[j][1]<=votes[j][0] and bias[i][j]>=0) or (votes[j][0]<=votes[j][1] and bias[i][j]<=0) or uniform(0,1)<accept_rate[j]:
						# 	truth[j][k]+=(answer[i][j][k]-bias[i][j])*quality[i]
						# else:
						# 	truth[j][k]+=(answer[i][j][k])*quality[i]
						if bias[i][j]*criminal_ratio[j]<0 or uniform(0,1)<accept_ratio[j]:
							truth[j][k]+=(answer[i][j][k]-bias[i][j])*quality[i]
						else:
							truth[j][k]+=(answer[i][j][k])*quality[i]
						acc_q+=quality[i]
				if acc_q==0:
					truth[j][k]=None
				else:
					truth[j][k]/=acc_q

		#Step 2: Joint Maximum Likelihood Estimation
		for i in range(0,n):
			for j in range(0,m):
				bias[i][j]=0.0
				acc_n=0
				for k in range(0,r[j]):
					if answer[i][j][k]!=None:
						bias[i][j]+=answer[i][j][k]-truth[j][k]
						acc_n+=1
				if acc_n==0:
					bias[i][j]=None
				else:
					bias[i][j]/=acc_n

		# print(bias)

		for i in range(0,n):
			sigma[i]=0.0
			acc_n=0
			for j in range(0,m):
				for k in range(0,r[j]):
					if answer[i][j][k]!=None:
						sigma[i]+=(answer[i][j][k]-truth[j][k]-bias[i][j])**2
						acc_n+=1
			sigma[i]=np.sqrt(sigma[i]/acc_n)

		# Step 3: Calculate worker quality
		if not VLDB:

			sum_q=0.0
			for i in range(0,len(quality)):
				quality[i]=1.0/sigma[i]**2
				sum_q+=quality[i]
			for i in range(0,len(quality)):
				quality[i]=quality[i]/sum_q+1E-10
			
		else:

			sum_quality=0.0
			for i in range(0,n):
				Ns=0
				up=1E-10
				down=1E-10
				for j in range(0,m):
					for k in range(0,r[j]):
						if answer[i][j][k]!=None:
							Ns+=1
							down+=(answer[i][j][k]-truth[j][k]-bias[i][j])**2
				up+=chi2.ppf(q=0.95,df=Ns)
				quality[i]=up/down+1E-10

			quality_sum=sum(quality)
			for i in range(0,n):
				quality[i]/=quality_sum

	return truth,bias,sigma,quality


