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
	TN=0
	TP=0
	FN=0
	FP=0
	up=0
	down=0
	positive=0
	negative=0
	criminal=0
	male_n=0
	for j in range(0,len(esti_truth)):
		for k in range(0,len(esti_truth[j])):
			if truth[j][k]!=None:
				if j in information['male']:
					if truth[j][k]==1:
						criminal+=1
					male_n+=1
			if j in information['male'] and esti_truth[j][k]!=None:
				if round(esti_truth[j][k])==1:
					positive+=1
				else:
					negative+=1
				if truth[j][k]==1 and round(esti_truth[j][k])==1:
					TP+=1
				if truth[j][k]==1 and round(esti_truth[j][k])==0:
					FP+=1
				if truth[j][k]==0 and round(esti_truth[j][k])==1:
					FN+=1
				if truth[j][k]==0 and round(esti_truth[j][k])==0:
					TN+=1
				down+=1
				up+=1-abs(esti_truth[j][k]-truth[j][k])
	# ret['male']=1.0*(positive)/(FP+FN+TP+TN)
	ret['male']=up/down
	print('Male criminal:',criminal,male_n)


	#P(acc|female)
	TN=0
	TP=0
	FN=0
	FP=0
	up=0
	down=0
	positive=0
	negative=0
	criminal=0
	female_n=0
	for j in range(0,len(esti_truth)):
		for k in range(0,len(esti_truth[j])):
			if truth[j][k]!=None:
				if j in information['female']:
					if truth[j][k]==1:
						criminal+=1
					female_n+=1
			if j in information['female'] and esti_truth[j][k]!=None:
				if round(esti_truth[j][k])==1:
					positive+=1
				else:
					negative+=1
				if truth[j][k]==1 and round(esti_truth[j][k])==1:
					TP+=1
				if truth[j][k]==1 and round(esti_truth[j][k])==0:
					FP+=1
				if truth[j][k]==0 and round(esti_truth[j][k])==1:
					FN+=1
				if truth[j][k]==0 and round(esti_truth[j][k])==0:
					TN+=1
				down+=1
				up+=1-abs(esti_truth[j][k]-truth[j][k])
	# ret['female']=1.0*(positive)/(FP+FN+TP+TN)
	ret['female']=up/down
	print('Female criminal:',criminal,female_n)

	count=0
	correct=0
	for j in range(0,len(esti_truth)):
		for k in range(0,len(esti_truth[j])):
			if esti_truth[j][k]!=None and round(esti_truth[j][k])==truth[j][k]:
				correct+=1
			count+=1

	print(correct,count)


	return ret['male'],ret['female'],ret['male']-ret['female'],1.0*correct/count


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

	FTI_truth,FTI_bias,FTI_sigma,FTI_quality=FTI(answer,VLDB=True,ground_truth=truth)
	TP,TN,FP,FN=confusion_matrix(esti_truth=FTI_truth,truth=truth)

	votes=[]
	for i in range(0,len(FTI_truth)):
		votes.append([0,0])
		for j in range(0,len(FTI_truth[i])):
			votes[-1][int(round(FTI_truth[i][j]))]+=1
	votes=np.array(votes)
	print(votes)
	exit()
	

	NTI_truth,NTI_bias,NTI_sigma,NTI_quality=NTI(answer,VLDB=True)
	TP,TN,FP,FN=confusion_matrix(esti_truth=NTI_truth,truth=truth)
	print(1.0*(TP+TN)/(TP+TN+FP+FN))

if __name__=='__main__':
	realworld_crime()