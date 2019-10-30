import numpy as np
from numpy.random import *
import sys,itertools,os
from scipy.stats import pearsonr
from scipy.stats import chi2
from math import exp
from copy import deepcopy
from truth_inference import FTI,NTI
from metric import *
from fair_truth_inference_bias import FTI_bias

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

	n=len(answer)
	m=len(answer[0])
	fraction=0.1
	selected=[]
	for i in range(0,int(n*fraction)):
		x=int(uniform(0,n))
		if x not in selected:
			selected.append(x)
	selected.sort()
	print(selected,len(selected))
	selected_answer=[]
	for item in selected:
		selected_answer.append(answer[item])

	from fair_truth_inference_bias import FTI_bias
	selected_truth,selected_bias,selected_sigma,selected_quality=FTI_bias(selected_answer,real_truth=truth)

	collected_answer=[]
	for i in range(0,len(answer)):
		collected_answer.append([])
		for j in range(0,len(answer[i])):
			collected_answer[-1].append([])
			for k in range(0,len(answer[i][j])):
				collected_answer[-1][-1].append(None)
	for i in range(0,len(selected)):
		collected_answer[selected[i]]=answer[selected[i]]

	collected_bias=[]
	for i in range(0,len(answer)):
		collected_bias.append([])
		for j in range(0,len(answer[i])):
			collected_bias[-1].append(None)
	for i in range(0,len(selected)):
		collected_bias[selected[i]]=selected_bias[i]

	print(collected_answer)

	collected_index=selected
	while True:
		if len(collected_index)==len(answer):
			break

		collected_truth,collected_bias,collected_sigma,collected_quality=FTI_bias(collected_answer,real_bias=collected_bias)

		TP,TN,FP,FN=confusion_matrix(esti_truth=collected_truth,truth=truth)
		print((TP+TN)/(TP+TN+FP+FN))
		print('\n')

		exit()





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