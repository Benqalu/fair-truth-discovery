
import numpy as np
from numpy.random import *
import sys,itertools,os
from scipy.stats import pearsonr
from scipy.stats import chi2
from math import exp
from copy import deepcopy

def confusion_matrix(inferred,truth,bound=0.5):
	TP=0
	TN=0
	FP=0
	FN=0
	total=0
	for i in range(0,len(inferred)):
		for j in range(0,len(inferred[i])):
			if inferred[i][j]==None or truth[i][j]==None:
				continue
			total+=1
			if inferred[i][j]>bound and truth[i][j]>bound:
				TP+=1
			if inferred[i][j]<bound and truth[i][j]<bound:
				TN+=1
			if inferred[i][j]>bound and truth[i][j]<bound:
				FP+=1
			if inferred[i][j]<bound and truth[i][j]>bound:
				FN+=1
	TP/=total
	TN/=total
	FP/=total
	FN/=total

	return TP,TN,FP,FN

def ranking(pmat,max_iter=100000):
	if type(pmat)==list:
		pmat=np.array(pmat)
	n=pmat.shape[0]
	wins=np.sum(pmat,axis=0)
	params=np.ones(n,dtype=float)
	for _ in range(max_iter):
		tiled=np.tile(params,(n,1))
		combined=1.0/(tiled+tiled.T)
		np.fill_diagonal(combined,0)
		nxt=wins/np.sum(combined,axis=0)
		nxt=nxt/np.mean(nxt)
		tmp=np.linalg.norm(nxt-params,ord=np.inf)
		if tmp<1e-5:
			return nxt
		params=nxt
	raise RuntimeError('did not converge')


def tau_distance(A,B):
	pairs=itertools.combinations(list(range(0,len(A))),2)
	distance=0
	for x,y in pairs:
		a=A[x]-A[y]
		b=B[x]-B[y]
		if (a*b<0):
			distance+=1
	n=len(A)
	n=n*(n-1)/2.0
	return 1.0*distance/n

def count_usable(a):
	c=0
	for item in a:
		if item!=None:
			c+=1
	return c

def matrix_add(A,B):
	if A==None or B==None:
		return None

	C=[[None for j in range(0,len(A[i]))] for i in range(0,len(A))]
	for i in range(0,len(A)):
		for j in range(0,len(A[i])):
			if A[i][j]==None or B[i][j]==None:
				C[i][j]==None
			else:
				C[i][j]=A[i][j]+B[i][j]
	return C

def user_profile_reader(fname):
	profile={}

	f=open(fname)
	f.readline()
	for row in f:
		ele=row.split(',')
		wid=ele[0].strip()
		age=int(ele[1])
		gender=ele[2].strip()
		if gender=='F':
			gender='woman'
		elif gender=='M':
			gender='man'
		else:
			continue
		race=ele[3].strip()
		if race!='multi' and race!='white':
			race='color'
		if race not in profile:
			profile[race]={}
		if gender not in profile[race]:
			profile[race][gender]=[]
		profile[race][gender].append(wid)
	f.close()

	for item in profile:
		for jtem in profile[item]:
			profile[item][jtem]=sorted(profile[item][jtem])

	return profile

def data_reader(fname):
	worker_id=[]
	res={}

	f=open(fname)
	f.readline()
	for row in f:
		elements=row.strip().split(',')
		for i in range(0,len(elements)):
			if i==0:
				worker_id.append(elements[i])
				res[worker_id[-1]]=[]
				continue
			e=elements[i]
			res[worker_id[-1]].append([])
			for s in e.strip():
				if s=='-':
					res[worker_id[-1]][-1].append(None)
				if s=='1':
					res[worker_id[-1]][-1].append(1)
				if s=='0':
					res[worker_id[-1]][-1].append(0)
	f.close()

	worker_id=sorted(worker_id)
	res_=[]
	for item in worker_id:
		res_.append(res[item])

	res=res_

	#------
	flipped=set([])
	flip_column=[0,4,7,9]
	for column_n in flip_column:
		r=len(res[0][column_n])
		flip=[uniform(0,1)<0.5 for i in range(0,r)]
		#print flip
		for i in range(0,len(res)):
			j=column_n
			for k in range(0,r):
				if res[i][j][k]!=None and flip[k]:
					res[i][j][k]=1-res[i][j][k]
					flipped.add((j,k))
	#------

	return worker_id,res,flipped

def clipped(x,a,b):
	if x<a:
		return a
	if x>b:
		return b
	return x

def data_generator(object_n,worker_n,category_n,compare_ratio=1.0,worker_portion=[0.1,0.9],skewness_right=0.00):

	object_score=uniform(0.1,0.9,object_n)

	ranking=np.flip(np.array(object_score).argsort())

	objects=[[] for i in range(0,category_n)]

	portion=uniform(worker_portion[0],worker_portion[1],worker_n)

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
			bias[i].append(uniform(-0.3,0.3)+skewness_right)
			# if uniform(0,1)<0.5:
			# 	bias[i].append(uniform(0.1,0.3)+skewness_right)
			# else:
			# 	bias[i].append(-uniform(0.1,0.3)+skewness_right)
	sigma=[]
	for i in range(0,worker_n):
		sigma.append(uniform(0.01,0.5))

	truth=[]
	for i in range(0,len(objects)):
		truth.append([])
		for j in range(0,len(objects[i])):
			a=objects[i][j][0]
			b=objects[i][j][1]
			truth[-1].append(1.0*object_score[b]/(object_score[a]+object_score[b]))

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
						answer[i][j][k]=clipped(truth[j][k]+bias[i][j]+normal(0,sigma[i]),0.0,1.0)

	return answer,truth,objects,ranking,bias,sigma

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

def FTI(answer,VLDB=False):

	n=len(answer)
	m=len(answer[0])
	r=[len(answer[0][j]) for j in range(0,m)]

	#Step 0: Give random initial numbers
	truth=[[0.0 for k in range(0,r[j])] for j in range(0,m)]
	bias=[[uniform(-0.1,0.1) for j in range(0,m)] for i in range(0,n)]
	sigma=[0.0 for i in range(0,n)]
	quality=[1.0/n for i in range(0,n)]
	last_truth=np.array([])

	while True:

		if len(last_truth)!=0:
			maxdiff=-1
			for j in range(0,m):
				for k in range(0,r[j]):
					if last_truth[j][k]!=None and truth[j][k]!=None: 
						maxdiff=max(maxdiff,abs(last_truth[j][k]-truth[j][k]))
			print(maxdiff)
			if maxdiff<1E-5 and maxdiff>=0:
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

def MAE(A,B):
	count=len(A)*1.0
	ret=0.0
	if type(A[0])==list:
		count*=len(A[0])
		for i in range(0,len(A)):
			for j in range(0,len(A[i])):
				ret+=abs(A[i][j]-B[i][j])
	else:
		for i in range(0,len(A)):
			ret+=abs(A[i]-B[i])
	return ret/count

def get_objects(fname):
	objects=[]
	f=open(fname)
	for row in f:
		objects.append([])
		eles=row.strip().split(',')
		for j in range(0,len(eles)):
			a=eles[j].split('-')
			b=int(a[1])-1
			a=int(a[0])-1
			objects[-1].append((a,b))
	f.close()
	return objects

def infer_ranking(truth,objects):

	objects_n=0
	for i in range(0,len(objects)):
		for j in range(0,len(objects[i])):
			if objects[i][j][0]+1>objects_n:
				objects_n=objects[i][j][0]+1
			if objects[i][j][1]+1>objects_n:
				objects_n=objects[i][j][1]+1

	p=[[0.5 for i in range(0,objects_n)] for j in range(0,objects_n)]

	for i in range(0,len(objects)):
		for j in range(0,len(objects[i])):
			p[objects[i][j][0]][objects[i][j][1]]=truth[i][j]
			p[objects[i][j][1]][objects[i][j][0]]=1-truth[i][j]

	return sorted(ranking(p).tolist(),reverse=True),np.flip(ranking(p).argsort())

def pearson(A,B):
	AA=[]
	BB=[]
	if type(A[0])==list:
		for item in A:
			AA+=item
		for item in B:
			BB+=item
	return pearsonr(AA,BB)[0]

def MAE(A,B):
	res=0.0
	count=0.0
	for i in range(0,len(A)):
		for j in range(0,len(A[i])):
			if A[i][j]==None or B[i][j]==None:
				continue
			count+=1
			res+=abs(A[i][j]-B[i][j])
	return res/count

def Synthetic():

	object_n=30
	worker_n=100
	category_n=5
	compare_ratio=1.0
	turn=20

	print('Number of objects:',object_n)
	print('Number of workers:',worker_n)
	print('Number of categories:',category_n)

	for c in reversed(list(range(1,11))):

		compare_ratio=0.1*c

		print('compare_ratio =',compare_ratio)

		result=[[],[],[]]
		error=[[],[],[]]

		for t in range(0,turn):

			# print('')
			# print('#'*30)
			# print('')

			# print('Compare ratio',compare_ratio)

			answer,truth,objects,true_ranking,bias,sigma=data_generator(object_n,worker_n,category_n,compare_ratio)
			# print('Data generated.\n')

			# print('True ranking:')
			# print(true_ranking)
			# print('')

			score,inferred_ranking=infer_ranking(truth=truth,objects=objects)
			# print('Inferred ranking:')
			tau=tau_distance(true_ranking,inferred_ranking)
			# print(inferred_ranking,tau0)
			# print('')
			result[0].append(tau)
			error[0].append(0.0)

			# print('Running normal truth inference...')
			score,NTI_ranking,NTI_truth=NTI_main(answer=answer,objects=objects,turn=3)
			NTI_tau=tau_distance(true_ranking,NTI_ranking)
			NTI_MAE=MAE(NTI_truth,truth)
			# print(NTI_ranking,tau1)
			#print 'MAE between truth:',MAE(truth,NTI_truth)
			# print('')
			result[1].append(NTI_tau)
			error[1].append(NTI_MAE)

			# print('Running fair truth inference...')
			score,FTI_ranking,FTI_truth=FTI_main(answer=answer,objects=objects,turn=3)
			FTI_tau=tau_distance(true_ranking,FTI_ranking)
			FTI_MAE=MAE(FTI_truth,truth)
			# print(FTI_ranking,tau2)
			#print 'MAE between truth:',MAE(truth,FTI_truth)
			# print('')
			result[2].append(FTI_tau)
			error[2].append(FTI_MAE)

			print(t,[sum(result[0])/(t+1),sum(result[1])/(t+1),sum(result[2])/(t+1)])

			# if tau1-tau2>0.15:
			# 	print truth
			# 	print bias
			# 	print sigma
			# 	print objects
			# 	print truth
			# 	exit()

		for i in range(0,3):
			result[i]=sum(result[i])/turn
			error[i]=sum(error[i])/turn

		print('*'*10,result,'*'*10)

		f=open('result/synthetic_n%d.txt'%worker_n,'a')
		f.write(str(compare_ratio)+'\t'+str(result)+'\t'+str(error)+'\n')
		f.close()

def mean_bias(bias):
	aggregated_count=[0.0 for i in range(0,len(bias[0]))]
	aggregated_bias=[0.0 for i in range(0,len(bias[0]))]
	for i in range(0,len(bias)):
		for j in range(0,len(bias[i])):
			if bias[i][j]==None:
				continue
			aggregated_count[j]+=1
			aggregated_bias[j]+=bias[i][j]
	for i in range(0,len(bias[0])):
		if aggregated_count[i]==0:
			aggregated_bias[i]=None
		else:
			aggregated_bias[i]/=aggregated_count[i]
	return aggregated_bias

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
			f.close()

	FTI_truth,FTI_bias,FTI_sigma,FTI_quality=FTI(answer,VLDB=True)

	votes=[]
	for j in range(0,len(truth)):
		votes.append([])
		for k in range(0,len(truth[j])):
			votes[-1].append([0,0])
			for i in range(0,len(answer)):
				if answer[i][j][k]!=None:
					votes[j][k][int(round(answer[i][j][k]))]+=1

	for j in range(0,len(truth)):
		for k in range(0,len(truth[j])):
			if truth[j][k]!=None:
				biases=[]
				for i in range(0,len(FTI_bias)):
					if answer[i][j][k]!=None:
						biases.append((answer[i][j][k],FTI_bias[i][j]))
				if truth[j][k]!=int(round(FTI_truth[j][k])):
					print((j,k),votes[j][k],truth[j][k],biases)


def stastic_debiasing():

	stastic_bias={}
	f=open('./realworld/crime/ibiasv.csv')
	for row in f:
		row=row.strip()
		if row[-1]==',':
			row=row[:-1]
		row=row.split(',')
		for i in range(1,len(row)):
			row[i]=eval(row[i])
			if row[i]!=None:
				row[i]=float(row[i])
		stastic_bias[row[0]]=row[1:]
	f.close()

	f=open('realworld/crime/truth.txt')
	truth=eval(f.readline())
	f.close()

	all_answer=[]
	all_workerid=[]
	fnames=os.listdir('realworld/crime')
	for fname in fnames:
		if 'crime' in fname:
			answer=[]
			print(fname)
			f=open('realworld/crime/'+fname)
			for row in f:
				e=row.strip().split('\t')
				all_answer.append(eval(e[1]))
				all_workerid.append(e[0])
			f.close()

	z={}
	for i in range(0,len(all_workerid)):
		z[all_workerid[i]]=all_answer[i]
	
	bias_sequence=[]
	for item in stastic_bias:
		for j in range(0,len(stastic_bias[item])):
			if stastic_bias[item][j]!=None:
				bias_sequence.append((item,j,abs(stastic_bias[item][j]),stastic_bias[item][j]))
	bias_sequence=sorted(bias_sequence,reverse=True,key=lambda bias_sequence:bias_sequence[2])

	i=-1
	while i<len(bias_sequence)-1:
		i+=1
		print(i,len(bias_sequence))
		wid=bias_sequence[i][0]
		cid=bias_sequence[i][1]
		bval=bias_sequence[i][3]

		print(bias_sequence[i],)

		if wid not in z:
			continue

		for k in range(0,len(z[wid][cid])):
			# if z[wid][cid][k]!=None:	#Delete
			# 	z[wid][cid][k]=None
			if z[wid][cid][k]!=None:	#Flip
				if bval*(z[wid][cid][k]-0.5)>0:
					z[wid][cid][k]=1-z[wid][cid][k]

		answer=[]
		for item in z:
			answer.append(z[item])

		esti_truth,bias,sigma,quality=NTI(answer)
		TPr,TNr,FPr,FNr=confusion_matrix(inferred=esti_truth,truth=truth)
		# print(TPr,TNr,FPr,FNr)
		print('Accuracy : %.4f'%((TPr+TNr)/(TPr+TNr+FPr+FNr)))#,'\t','Precision :',TPr/(TPr+FPr),'\t','Recall :',TPr/(TPr+FNr))

if __name__=='__main__':
	realworld_crime()
	# Synthetic()
	# stastic_debiasing()

	# answer,truth,objects,true_ranking,bias,sigma=data_generator(object_n=6,worker_n=4,category_n=2,worker_portion=[1.0,1.0],compare_ratio=1.0)
	
	# print('Truth',truth)
	# print('Stddev',sigma)
	# print('Bias',bias,'\n')

	# fti_truth,fti_bias,fti_sigma,fti_quality=FTI(answer)
	
	# print(MAE(fti_truth,truth),'\n')

	# nti_truth,_,nti_sigma,nti_quality=NTI(answer,VLDB=True)

	# print(MAE(nti_truth,truth),'\n')

