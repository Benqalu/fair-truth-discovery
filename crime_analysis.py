import numpy as np
from numpy.random import *
import sys,itertools,os
from scipy.stats import pearsonr
from scipy.stats import chi2
from math import exp,log
from copy import deepcopy
from truth_inference import FTI,NTI
from metric import *
from fair_truth_inference_bias import FTI_bias

class Metric(object):

	def __init__(self,ground_truth):
		self.ground_truth=ground_truth

	def accuracy(self,truth,binary=True,threshold=0.5):
		down=0.0
		up=0.0
		for i in range(0,len(truth)):
			for j in range(0,len(truth[i])):
				if truth[i][j]!=None:
					down+=1
					if truth[i][j]>=threshold and self.ground_truth[i][j]==1:
						up+=1
					if truth[i][j]<threshold and self.ground_truth[i][j]==0:
						up+=1
		return up/down

	def disparity(self,truth,binary=True,threshold=0.5):

		m=len(self.ground_truth)

		disparity=np.array([0.0 for i in range(0,m)])

		disparity_self=[[0.0,0.0] for i in range(0,m)]

		for j in range(0,m):
			for k in range(0,len(truth[j])):
				if truth[j][k]==None:
					continue
				disparity_self[j][1]+=1
				if truth[j][k]>threshold:
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
					if truth[j][k]>threshold:
						disparity_other[j_][0]+=1
		for j in range(0,m):
			disparity_other[j]=disparity_other[j][0]/disparity_other[j][1]

		disparity=[]
		for j in range(0,m):
			disparity.append(disparity_self[j]-disparity_other[j])
		disparity=np.array(disparity)

		return disparity

def realworld_crime(A):

	f=open('realworld/crime/truth.txt')
	truth=eval(f.readline())
	f.close()

	answer=[]
	workerid=[]

	f=open('./realworld/crime/resultsv_noage.txt')
	f.readline()
	for row in f:
		answer.append([])

		row=row.strip()
		if row[-1]=='|':
			row=row[:-1]
		row=row.split('|')
		for i in range(1,len(row)):
			item=row[i]
			answer[-1].append([])
			for e in item:
				if e=='-':
					answer[-1][-1].append(None)
				else:
					answer[-1][-1].append(int(e))
		workerid.append(row[0])
	f.close()

	if A=='race':
		for i in range(0,len(answer)):
			answer[i][0]+=answer[i][1]
			answer[i][3]+=answer[i][2]
			answer[i][5]+=answer[i][4]
			del answer[i][7]
			del answer[i][5]
			del answer[i][4]
			del answer[i][2]
			del answer[i][1]
		truth[0]+=truth[1]
		truth[3]+=truth[2]
		truth[5]+=truth[4]
		del truth[7]
		del truth[5]
		del truth[4]
		del truth[2]
		del truth[1]

	if A=='gender':
		for i in range(0,len(answer)):
			answer[i][1]=answer[i][1]+answer[i][2]+answer[i][4]
			answer[i][0]=answer[i][0]+answer[i][3]+answer[i][6]
			answer[i]=answer[i][:2]
		truth[1]=truth[1]+truth[2]+truth[4]
		truth[0]=truth[0]+truth[3]+truth[6]
		truth=truth[:2]

	return answer,truth,workerid



def experiment_results_accuracy_disparity(theta,ground_truth,turn=100):

	metric=Metric(ground_truth)

	average_accuracy=0.0
	average_disparity=np.array([0.0 for j in range(0,len(ground_truth))])

	for t in range(0,turn):

		#Round 1
		truth,bias,sigma,quality=FTI(answer)
		#Round 2
		truth,bias,sigma,quality=FTI(
			answer,
			init_truth=truth,
			init_bias=bias,
			init_sigma=sigma,
			init_quality=quality,
			theta=theta
		)

		average_accuracy+=metric.accuracy(truth=truth)
		average_disparity+=metric.disparity(truth=truth)

		print(average_accuracy/(t+1),average_disparity/(t+1))

	return average_accuracy/turn,(average_disparity/turn).tolist()


if __name__=='__main__':

	A='race'

	answer,truth,workerid=realworld_crime(A=A)

	for theta in range(1,101):
		print('theta =',theta*0.01)
		accuracy,disparity=experiment_results_accuracy_disparity(theta=0.01*theta,ground_truth=truth,turn=10)
		print('')

		f=open('result/theta_%s.txt'%A,'a')
		f.write(str(0.01*theta)+'\t'+str(accuracy)+'\t'+str(disparity)+'\n')
		f.close()

	exit()

	f=open('bias_%s.txt'%A,'w')
	for i in range(0,len(bias_fti)):
		f.write(str(workerid[i])+'\t')
		for j in range(0,len(bias_fti[i])):
			f.write(str(bias_fti[i][j])+'\t')
		f.write('\n')
	f.close()
