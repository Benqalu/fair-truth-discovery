import sys
import numpy as np
from random import *
from math import exp
from copy import deepcopy
from collections import OrderedDict
from data_reader import *
from metrics import *

def FairTD_In(answer,theta=0.01):

	n=len(answer)
	m=len(answer[0])
	r=[len(answer[0][j]) for j in range(0,m)]

	truth=[[uniform(0.1,0.9) for k in range(0,r[j])] for j in range(0,m)]
	bias=[[uniform(-0.1,0.1) for j in range(0,m)] for i in range(0,n)]
	sigma=[0.0 for i in range(0,n)]
	quality=[1.0/n for i in range(0,n)]

	last_truth=np.array([])

	disparity=np.array([0.0 for i in range(0,m)])

	maxdiff=-1

	repeat_times=-1

	while True:

		repeat_times+=1

		if theta!=None:
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

			print('Iterated %d times, current disparity:'%repeat_times,disparity.tolist(),'\r',end=' ')
			sys.stdout.flush()

		if len(last_truth)!=0:
			maxdiff=-1
			for j in range(0,m):
				for k in range(0,r[j]):
					if last_truth[j][k]!=None and truth[j][k]!=None: 
						maxdiff=max(maxdiff,abs(last_truth[j][k]-truth[j][k]))
			if maxdiff<1E-2 and maxdiff>=0:
				if theta==None or (theta!=None and max(abs(disparity))<theta):
					break
				else:
					pass
			else:
				pass
		if theta!=None:
			rejection_list=OrderedDict()
			sum_score=0.0
			for i in range(0,n):
				for j in range(0,m):
					z=disparity[j]*bias[i][j]
					if z<0:
						rejection_list[(i,j)]=-z
						sum_score+=-z
			p=uniform(0,1)
			acc=0.0
			selected=None
			for item in rejection_list:
				rejection_list[item]/=sum_score
				acc+=rejection_list[item]
				if acc>=p:
					selected=item
					break

		last_truth=deepcopy(truth)
		for j in range(0,m):
			for k in range(0,r[j]):
				truth[j][k]=0.0
				acc_q=0.0
				for i in range(0,n):
					if answer[i][j][k]!=None:
						if theta!=None:
							# if bias[i][j]*disparity[j]>0 or uniform(0,1)>abs(disparity[j]):# or max(abs(disparity))<theta:
							if (i,j)!=selected or max(abs(disparity))<theta:
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
						sigma[i]+=(answer[i][j][k]-truth[j][k]-bias[i][j])**2
						acc_n+=1
			sigma[i]=np.sqrt(sigma[i]/acc_n)

		sum_q=0.0
		for i in range(0,len(quality)):
			quality[i]=1.0/sigma[i]**2
			sum_q+=quality[i]
		for i in range(0,len(quality)):
			quality[i]=quality[i]/sum_q+1E-10

	return truth,bias,sigma
