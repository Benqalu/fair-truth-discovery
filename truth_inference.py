import numpy as np
from random import *
import sys,itertools,os
from scipy.stats import pearsonr
from scipy.stats import chi2
from math import exp
from copy import deepcopy
from collections import OrderedDict

def MV(answer):
	truth=[]
	for j in range(len(answer[0])):
		truth.append([])
		for k in range(len(answer[0][j])):
			truth[-1].append([0,0])
	for i in range(0,len(answer)):
		for j in range(0,len(answer[i])):
			for k in range(0,len(answer[i][j])):
				if answer[i][j][k]!=None:
					truth[j][k][answer[i][j][k]]+=1
	for j in range(0,len(truth)):
		for k in range(0,len(truth[j])):
			if truth[j][k][0]>truth[j][k][1]:
				truth[j][k]=0
			elif truth[j][k][0]<truth[j][k][1]:
				truth[j][k]=1
			elif truth[j][k][0]==truth[j][k][1]:
				truth[j][k]=int(uniform(0,1)>=0.5)
			else:
				truth[j][k]=None
	return truth

def NTI(answer,CATD=True):

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