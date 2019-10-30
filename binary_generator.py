import numpy as np
from numpy.random import *

from FTI import NTI,confusion_matrix

def clipped(x,a,b):
	if x<a:
		return a
	if x>b:
		return b
	return x

def data_generator(object_n,worker_n,category_n,compare_ratio=1.0,skewness_right=0.00):

	object_score=uniform(0,1,object_n)

	ranking=np.flip(np.array(object_score).argsort())

	objects=[[] for i in range(0,category_n)]

	portion=poisson(0.9,worker_n)*0.01

	task_list=[]
	for i in range(0,object_n):
		for j in range(0,i):
			if i==j or uniform(0,1)>compare_ratio:
				continue
			task_list.append((i,j))

	shuffle(task_list)
	r=poisson(int(len(task_list)/category_n),category_n).tolist()
	p=0
	l=len(task_list)
	for i in range(0,len(r)):
		for j in range(0,r[i]):
			objects[i].append(task_list[p])
			p+=1
			if p==l:
				break
		if p==l:
			break

	bias=[]
	for i in range(0,worker_n):
		bias.append([])
		for j in range(0,category_n):
			bias[i].append(0.0)
			# if uniform(0,1)<0.5:
				# bias[i].append(poisson(10)*0.01)
			# else:
				# bias[i].append(-poisson(10)*0.01)
	sigma=[]
	for i in range(0,worker_n):
		sigma.append(poisson(100)*0.01)

	truth=[]
	for i in range(0,len(objects)):
		truth.append([])
		for j in range(0,len(objects[i])):
			a=objects[i][j][0]
			b=objects[i][j][1]
			truth[-1].append(int(round(1.0*object_score[b]/(object_score[a]+object_score[b]))))

	count=[]
	for i in range(0,len(objects)):
		count.append([])
		for j in range(0,len(objects[i])):
			count[i].append(0)

	answer=[]
	for i in range(0,worker_n):
		answer.append([])
		for j in range(0,category_n):
			answer[i].append([])
			for k in range(0,len(truth[j])):
				answer[i][j].append(None)

	for j in range(0,category_n):
		for k in range(0,len(truth[j])):
				l=list(range(0,worker_n))
				shuffle(l)
				for i in l:
					if uniform(0,1)<portion[i] or count[j][k]==0:
						answer[i][j][k]=int(round(clipped(truth[j][k]+bias[i][j]+normal(0,sigma[i]),0.0,1.0)))

	return answer,truth,objects,ranking,bias,sigma

def MajorityVoting(answer):
	votes=[]
	for i in range(0,len(answer[0])):
		votes.append([])
		for j in range(0,len(answer[0][i])):
			votes[-1].append([0,0])

	for i in range(0,len(answer)):
		for j in range(0,len(answer[i])):
			for k in range(0,len(answer[i][j])):
				if answer[i][j][k]!=None:
					votes[j][k][answer[i][j][k]]+=1

	for j in range(0,len(answer[0])):
		for k in range(0,len(answer[0][j])):
			votes[j][k]=np.argmax(votes[j][k])

	return votes

if __name__=='__main__':

	turn=100

	res=np.array([0.0,0.0])

	for t in range(0,turn):

		print(t)

		answer,truth,objects,ranking,bias,sigma=data_generator(object_n=50,worker_n=30,category_n=10)	

		esti_truth,_,_,esti_quality=NTI(answer,VLDB=False)

		TP,TN,FP,FN=confusion_matrix(inferred=esti_truth,truth=truth)
		ti_acc=(TP+TN)/(TP+TN+FP+FN)
		print('TruthInference Accuracy :',ti_acc)

		esti_voting=MajorityVoting(answer)
		TP,TN,FP,FN=confusion_matrix(inferred=esti_voting,truth=truth)
		mv_acc=(TP+TN)/(TP+TN+FP+FN)
		print('MajorityVoting Accuracy :',mv_acc)

		res+=np.array([ti_acc,mv_acc])

	print(res/turn)