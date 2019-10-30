import numpy as np
from numpy.random import *
import sys,itertools,os
from scipy.stats import pearsonr
from scipy.stats import chi2
from math import exp
from copy import deepcopy
from truth_inference import FTI,NTI
from metric import *

def measure_fair_acc(esti_truth,truth):

	information={}
	information['male']=[1,2,4,5,7]
	information['female']=[0,3,6]

	ret={}

	#P(acc|male)
	up=0.0
	down=0.0
	for j in range(0,len(esti_truth)):
		if j not in information['male']:
			continue
		for k in range(0,len(esti_truth[j])):
			if truth[j][k]!=None:
				down+=1
				up+=1-abs(esti_truth[j][k]-truth[j][k])
	# ret['male']=1.0*(positive)/(FP+FN+TP+TN)
	ret['male']=up/down

	#P(acc|female)
	up=0.0
	down=0.0
	for j in range(0,len(esti_truth)):
		if j not in information['female']:
			continue
		for k in range(0,len(esti_truth[j])):
			if truth[j][k]!=None:
				down+=1
				up+=1-abs(esti_truth[j][k]-truth[j][k])
	ret['female']=up/down

	return ret['male'],ret['female'],ret['male']-ret['female']


def realworld_crime():

	f=open('realworld/crime/truth.txt')
	truth=eval(f.readline())
	f.close()

	del truth[7]
	del truth[6]
	del truth[5]
	del truth[4]

	answer=[]
	label=[]
	workerid=[]

	fnames=os.listdir('realworld/crime')
	for fname in fnames:
		if 'crime' in fname:
			this_answer=[]
			f=open('realworld/crime/'+fname)
			for row in f:
				e=row.strip().split('\t')
				this_answer.append(eval(e[1]))
				answer.append(eval(e[1]))
				workerid.append(e[0])
				label.append(fname.split('.')[0][6:])
				
				del answer[-1][7]
				del answer[-1][6]
				del answer[-1][5]
				del answer[-1][4]
				
			f.close()

	# FTI_truth,FTI_bias,FTI_sigma,FTI_quality=FTI(answer,VLDB=True,with_fairness=True)
	# TP,TN,FP,FN=confusion_matrix(esti_truth=FTI_truth,truth=truth)

	NTI_truth,NTI_bias,NTI_sigma,NTI_quality=NTI(answer,VLDB=False)
	TP,TN,FP,FN=confusion_matrix(esti_truth=NTI_truth,truth=truth)
	print((TP+TN)/(TP+TN+FP+FN))
	print(measure_fair_acc(esti_truth=NTI_truth,truth=truth))
	print('\n')

	FTI_truth,FTI_bias,FTI_sigma,FTI_quality=FTI(answer,VLDB=False,with_fairness=False)
	TP,TN,FP,FN=confusion_matrix(esti_truth=FTI_truth,truth=truth)
	print((TP+TN)/(TP+TN+FP+FN))
	print(measure_fair_acc(esti_truth=FTI_truth,truth=truth))
	print('\n')


if __name__=='__main__':
	realworld_crime()