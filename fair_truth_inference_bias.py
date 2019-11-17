import numpy as np
from numpy.random import *
import sys,itertools,os
from scipy.stats import pearsonr
from scipy.stats import chi2
from math import exp
from copy import deepcopy
from metric import *

def FTI_bias(answer):

	n=len(answer)
	m=len(answer[0])
	r=[len(answer[0][j]) for j in range(0,m)]

	#Step 0: Give random initial numbers
	truth=[[uniform(0.1,0.9) for k in range(0,r[j])] for j in range(0,m)]
	bias=[[uniform(-0.1,0.1) for j in range(0,m)] for i in range(0,n)]
	sigma=[0.0 for i in range(0,n)]
	quality=[1.0/n for i in range(0,n)]
	last_truth=np.array([])

	last_truth=[]

	first=True

	while True:

		if len(last_truth)!=0:
			maxdiff=-1
			for j in range(0,m):
				for k in range(0,r[j]):
					if last_truth[j][k]!=None and truth[j][k]!=None: 
						maxdiff=max(maxdiff,abs(last_truth[j][k]-truth[j][k]))
			# print(maxdiff)
			if maxdiff<1E-5 and maxdiff>=0:
				break
			else:
				pass

		last_truth=deepcopy(truth)


		#Step 0: Calculate disparity offset
		if first:
			psi=[]
			phi=[]
			for j in range(0,len(answer[0])):
				up=0
				down=0
				for i in range(0,len(answer)):
					if item==None:
						continue
					for item in answer[i][j]:
						down+=1
					if item==1:
						up+=1
				psi.append(x/y)

				up=0
				down=0
				



		#Step 1: Estimate Truth
		for j in range(0,m):
			for k in range(0,r[j]):
				truth[j][k]=0.0
				acc_q=0.0
				for i in range(0,n):
					if answer[i][j][k]!=None:
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


