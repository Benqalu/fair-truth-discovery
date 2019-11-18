import numpy as np
from numpy.random import *
import sys,itertools,os
from scipy.stats import pearsonr
from scipy.stats import chi2
from math import exp
from copy import deepcopy

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

def FTI(answer,VLDB=False,init_bias=None,init_truth=None,init_sigma=None,init_quality=None,theta=0.01):
	# answer only, no reject
	n=len(answer)
	m=len(answer[0])
	r=[len(answer[0][j]) for j in range(0,m)]

	#Step 0: Give random initial numbers
	if init_truth != None:
		truth=init_truth
	else:
		truth=[[uniform(0.1,0.9) for k in range(0,r[j])] for j in range(0,m)]
	if init_bias != None:
		bias=init_bias
	else:
		bias=[[uniform(-0.1,0.1) for j in range(0,m)] for i in range(0,n)]
	if init_sigma != None:
		sigma=init_sigma
	else:
		sigma=[0.0 for i in range(0,n)]
	if init_quality != None:
		quality=init_quality
	else:
		quality=[1.0/n for i in range(0,n)]

	# print(sigma)

	last_truth=np.array([])

	disparity=np.array([0.0 for i in range(0,m)])

	while True:

		if init_truth!=None:
			disparity_self=[[0.0,0.0] for i in range(0,m)]
			for j in range(0,m):
				for k in range(0,len(truth[j])):
					if truth[j][k]==None:
						continue
					disparity_self[j][1]+=1
					if truth[j][k]>0.5:
						disparity_self[j][0]+=1
			for j in range(0,m):
				disparity_self[j]=disparity_self[j][0]/disparity_self[j][1]

			disparity_other=[[0.0,0.0] for i in range(0,m)]
			for j_ in range(0,m):
				for j in range(0,m):
					if j==j_:
						continue
					for k in range(0,len(truth[j])):
						if truth[j][k]==None:
							continue
						disparity_other[j_][1]+=1
						if truth[j][k]>0.5:
							disparity_other[j_][0]+=1
			for j in range(0,m):
				disparity_other[j]=disparity_other[j][0]/disparity_other[j][1]

			disparity=[]
			for j in range(0,m):
				disparity.append(disparity_self[j]-disparity_other[j])
			disparity=np.array(disparity)

			print(disparity.tolist(),'\r',end=' ')
			sys.stdout.flush()

		if len(last_truth)!=0:
			maxdiff=-1
			for j in range(0,m):
				for k in range(0,r[j]):
					if last_truth[j][k]!=None and truth[j][k]!=None: 
						maxdiff=max(maxdiff,abs(last_truth[j][k]-truth[j][k]))
			# print(maxdiff,disparity)
			if maxdiff<1E-2 and maxdiff>=0 and max(abs(disparity))<theta:
				break
			else:
				pass

		last_truth=deepcopy(truth)

		#Step 1: Estimate Truth
		for j in range(0,m):
			for k in range(0,r[j]):
				truth[j][k]=0.0
				acc_q=0.0
				for i in range(0,n):
					if answer[i][j][k]!=None:
						if init_truth!=None and init_bias!=None:
							if bias[i][j]*disparity[j]>0 or uniform(0,1)>abs(disparity[j]):
								truth[j][k]+=(answer[i][j][k]-bias[i][j])*quality[i]
							else:
								truth[j][k]+=answer[i][j][k]*quality[i]
						else:
							truth[j][k]+=(answer[i][j][k]-bias[i][j])*quality[i]
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

		for i in range(0,n):
			sigma[i]=0.0
			acc_n=0
			for j in range(0,m):
				for k in range(0,r[j]):
					if answer[i][j][k]!=None:
						# print(answer[i][j][k])
						# print(truth[j][k])
						# print(bias[i][j])
						sigma[i]+=(answer[i][j][k]-truth[j][k]-bias[i][j])**2
						# print(sigma[i])
						acc_n+=1
			sigma[i]=np.sqrt(sigma[i]/acc_n)

		# print(sigma)

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
