import numpy as np
# from numpy.random import *
from random import uniform,gauss
from majority_voting import majority_voting
from truth_inference import FTI,NTI

def clipped(x,a,b):
	if x<a:
		return a
	if x>b:
		return b
	return x

def accuracy(esti_truth,gtruth):
	up=0
	down=0
	for i in range(len(esti_truth)):
		for j in range(len(esti_truth[i])):
			if (esti_truth[i][j]>=0.5 and gtruth[i][j]>=0.5) or (esti_truth[i][j]<0.5 and gtruth[i][j]<0.5):
				up+=1
			down+=1
	return up/down

def disparity(truth,ground_truth,binary=True,threshold=0.5):

	m=len(ground_truth)

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

def data_generator(task_n,worker_n,category_n,ratio=0.2):

	category_centers=[uniform(0.3,0.7) for i in range(category_n)]

	truth=[]
	for j in range(category_n):
		truth.append([])
		for k in range(task_n):
			if uniform(0,1)>=category_centers[j]:
				truth[-1].append(0)
			else:
				truth[-1].append(1)

	sigma=[]
	for i in range(worker_n):
		sigma.append(uniform(0.01,6.0))

	bias=[]
	for i in range(worker_n):
		bias.append([])
		for j in range(0,category_n):
			bias[-1].append(uniform(-0.5,0.5))

	ans_each_cat=int(round(task_n*ratio))

	answer=[]
	for i in range(worker_n):
		answer.append([])
		for j in range(category_n):
			answer[-1].append([])
			for k in range(task_n):
				answer[-1][-1].append(None)
			for z in range(ans_each_cat):
				k=(i*ans_each_cat+z)%task_n
				ans=int(round(clipped(gauss(mu=truth[j][k]+bias[i][j],sigma=sigma[i]),0,1)))
				answer[-1][-1][k]=ans
	
	return answer,truth



if __name__=='__main__':

	# task_n=10000
	# worker_n=100
	# category_n=2

	# while True:

	# 	answer,truth=data_generator(
	# 		task_n=task_n,
	# 		worker_n=worker_n,
	# 		category_n=category_n,
	# 		ratio=0.10
	# 	)		

	# 	truth_nti,_,_,_=NTI(answer)
	# 	disp=disparity(truth=truth_nti,ground_truth=truth)
	# 	acc_nti=accuracy(esti_truth=truth_nti,gtruth=truth)

	# 	print('Accuracy_NTI =',acc_nti,disp)

	# 	if abs(disp[0])>0.10 and acc_nti<0.82:
	# 		esti_truth,esti_bias,sigma,quality=FTI(answer)
	# 		acc_fti_1=accuracy(esti_truth=esti_truth,gtruth=truth)
	# 		# esti_truth,esti_bias,sigma,quality=FTI(answer,init_bias=esti_bias,init_truth=esti_truth,init_sigma=sigma,init_quality=quality,theta=0.01)
	# 		# acc_fti_2=accuracy(esti_truth=esti_truth,gtruth=truth)
	# 		print('Accuracy FTI =',acc_fti_1)
	# 		if acc_fti_1>acc_nti:
	# 			break

	# f=open('synthetic_worker%d_task%d_category%d.txt'%(worker_n,task_n,category_n),'w')
	# f.write(str(answer)+'\n')
	# f.write(str(truth))
	# f.close()

	f=open('synthetic_worker100_task10000_category2.txt')
	answer=eval(f.readline())
	truth=eval(f.readline())
	f.close()

	esti_truth,esti_bias,sigma,quality=NTI(answer,VLDB=True)
	acc_fti_1=accuracy(esti_truth=esti_truth,gtruth=truth)
	print(acc_fti_1)

	# esti_truth,esti_bias,sigma,quality=FTI(answer)
	# acc_fti_1=accuracy(esti_truth=esti_truth,gtruth=truth)
	# print(acc_fti_1)


	# esti_truth,esti_bias,sigma,quality=FTI(answer,init_bias=esti_bias,init_truth=esti_truth,init_sigma=sigma,init_quality=quality,theta=0.01)
	# acc_fti_2=accuracy(esti_truth=esti_truth,gtruth=truth)
	# print(acc_fti_2)
