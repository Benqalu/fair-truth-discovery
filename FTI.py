from __future__ import division
import numpy as np
from numpy.random import *
import sys,itertools,os
from scipy.stats import pearsonr
from scipy.stats import chi2
from math import exp

def confusion_matrix(inferred,truth):
	TP=0
	TN=0
	FP=0
	FN=0
	total=0
	for i in range(0,len(inferred)):
		for j in range(0,len(inferred[i])):
			total+=1
			if inferred[i][j]==1 and truth[i][j]==1:
				TP+=1
			if inferred[i][j]==0 and truth[i][j]==0:
				TN+=1
			if inferred[i][j]==1 and truth[i][j]==0:
				FP+=1
			if inferred[i][j]==0 and truth[i][j]==1:
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
	for _ in xrange(max_iter):
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
	pairs=itertools.combinations(range(0,len(A)),2)
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

def data_generator(object_n,worker_n,category_n,compare_ratio=1.0,skewness_right=0.1):

	object_score=uniform(0,1,object_n)

	ranking=np.flip(np.array(object_score).argsort())

	objects=[[] for i in range(0,category_n)]

	portion=uniform(0.1,0.3,worker_n)

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
			# bias[i].append(0.1)
			bias[i].append(normal(skewness_right,0.15))

	sigma=[]
	for i in range(0,worker_n):
		sigma.append(uniform(0.01,0.2))

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
		answer[i].append([])
		for k in range(0,len(truth[j])):
				l=range(0,worker_n)
				shuffle(l)
				for i in l:
					if uniform(0,1)<portion[i] or count[j][k]==0:
						answer[i][j][k]=clipped(truth[j][k]+bias[i][j]+normal(0,sigma[i]),0.0,1.0)

	return answer,truth,objects,ranking,bias,sigma

def NTI(answer):

	n=len(answer)
	m=len(answer[0])
	r=[len(answer[0][j]) for j in range(0,m)]

	truth=[[uniform(0.49,0.51) for k in range(0,r[j])] for j in range(0,m)]
	sigma=[uniform(0,0.1) for i in range(0,n)]
	quality=[1.0/n for i in range(0,n)]

	last_truth=np.array([])

	while True:

		if last_truth.size!=0:
			maxdiff=0.0
			for j in range(0,m):
				for k in range(0,r[j]):
					maxdiff=max(maxdiff,abs(last_truth[j][k]-truth[j][k]))
			if maxdiff<1E-5:
				break
			else:
				pass
		last_truth=np.copy(truth)

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

		# quality=np.array(sigma)**2+1E-10
		# quality/=quality.sum()

		sum_quality=1E-10
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
			quality[i]=up/down

		for i in range(0,n):
			quality[i]/=sum(quality)

	return truth,None,None,quality

def FTI(answer):
	n=len(answer)
	m=len(answer[0])
	r=[len(answer[0][j]) for j in range(0,m)]

	#Step 0: Give random initial numbers
	truth=[[uniform(0,1) for k in range(0,r[j])] for j in range(0,m)]
	bias=[[uniform(-0.1,0.1) for j in range(0,m)] for i in range(0,n)]
	sigma=[uniform(0,0.2) for i in range(0,n)]
	quality=[1.0/n for i in range(0,n)]
	last_truth=np.array([])

	while True:

		if last_truth.size!=0:
			maxdiff=-1
			for j in range(0,m):
				for k in range(0,r[j]):
					if last_truth[j][k]!=None and truth[j][k]!=None: 
						maxdiff=max(maxdiff,abs(last_truth[j][k]-truth[j][k]))
			if maxdiff<1E-10 and maxdiff>=0:
				break
			else:
				pass
		last_truth=np.copy(truth)
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

		for i in range(0,n):
			sigma[i]=0.0
			acc_n=0
			for j in range(0,m):
				for k in range(0,r[j]):
					if answer[i][j][k]!=None:
						sigma[i]+=(answer[i][j][k]-truth[j][k]-bias[i][j])**2
						acc_n+=1
			sigma[i]=np.sqrt(sigma[i]/acc_n)

		#Step 3: Calculate worker quality
		#Option 1
		# quality=np.array(sigma)**2
		# quality/=quality.sum()

		#Option 2
		sum_quality=1E-10
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
			quality[i]=up/down

		sum_quality=sum(quality)
		for i in range(0,n):
			quality[i]/=sum_quality

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

def FTI_main(answer=None,objects=None,turn=1):

	total_bias=None
	total_truth=None

	for _ in range(0,turn):

		if objects==None:
			worker_id,answer,flipped=data_reader('results.csv')
			profile=user_profile_reader('demo.csv')

		truth,bias,sigma,quality=FTI(answer)

		if objects==None:
			for j in range(0,len(truth)):
				for k in range(0,len(truth[j])):
					if (j,k) in flipped:
						truth[j][k]=1-truth[j][k]

		if total_bias==None:
			total_bias=bias
		else:
			total_bias=matrix_add(total_bias,bias)

		if total_truth==None:
			total_truth=truth
		else:
			total_truth=matrix_add(total_truth,truth)

		print _+1,'/',turn,'\r',
		sys.stdout.flush()

	f=open('result/truth_FTI.txt','w')
	for j in range(0,len(total_truth)):
		for k in range(0,len(total_truth[j])):
			total_truth[j][k]/=turn
			f.write(str(total_truth[j][k]/turn)+'\t')
		f.write('\n')
	f.close()

	if objects==None:
		objects=get_objects('task.txt')
	score,ranking=infer_ranking(truth=truth,objects=objects)
	
	return score,ranking,total_truth

	exit()

	bias=total_bias
	for i in range(0,len(bias)):
		for j in range(0,len(bias[i])):
			if bias[i][j]!=None:
				bias[i][j]/=turn

	f=open('crowdsource_bias.txt','w')
	for i in range(0,len(bias)):
		f.write(worker_id[i]+'\t')
		for j in range(0,len(bias[i])):
			if count_usable(answer[i][j])>=3:
				f.write(str(bias[i][j])+'\t')
			else:
				f.write(str(None)+'\t')
				bias[i][j]=None
		f.write('\n')
	f.close()

	dict_bias={}
	for i in range(0,len(worker_id)):
		item=worker_id[i]
		dict_bias[item]=bias[i]

	for race in profile:
		for gender in profile[race]:
			f=open('./result/bias_%s_%s.txt'%(race,gender),'w')
			for wid in profile[race][gender]:
				f.write(wid+'\t')
				for item in dict_bias[wid]:
					f.write(str(item)+'\t')
				f.write('\n')
			f.close()

def NTI_main(answer=None,objects=None,turn=1):

	total_truth=None

	for _ in range(0,turn):

		# worker_id,answer=data_reader('noceo.csv')

		if objects==None:
			worker_id,answer,flipped=data_reader('results.csv')
			profile=user_profile_reader('demo.csv')

		truth,bias,sigma,quality=NTI(answer)

		if objects==None:
			for j in range(0,len(truth)):
				for k in range(0,len(truth[j])):
					if (j,k) in flipped:
						truth[j][k]=1-truth[j][k]

		if total_truth==None:
			total_truth=truth
		else:
			total_truth=matrix_add(total_truth,truth)

		print _+1,'/',turn,'\r',
		sys.stdout.flush()

	f=open('result/truth_NTI.txt','w')
	for j in range(0,len(total_truth)):
		for k in range(0,len(total_truth[j])):
			total_truth[j][k]/=turn
			f.write(str(total_truth[j][k]/turn)+'\t')
		f.write('\n')
	f.close()

	if objects==None:
		objects=get_objects('task.txt')
	score,ranking=infer_ranking(truth=truth,objects=objects)
	
	return score,ranking,total_truth

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
	category_n=10
	compare_ratio=1.0


	print 'Number of objects:',object_n
	print 'Number of workers:',worker_n
	print 'Number of categories:',category_n

	for c in reversed(range(1,11)):

		compare_ratio=0.1*c

		print 'compare_ratio =',compare_ratio

		result=[[],[],[]]

		for t in range(0,20):

			print ''
			print '#'*30
			print ''

			print 'turn =',t

			print 'Compare ratio',compare_ratio

			answer,truth,objects,true_ranking,bias,sigma=data_generator(object_n,worker_n,category_n,compare_ratio)
			print 'Data generated.\n'

			print 'True ranking:'
			print true_ranking
			print ''

			score,inferred_ranking=infer_ranking(truth=truth,objects=objects)
			print 'Inferred ranking:'
			tau0=tau_distance(true_ranking,inferred_ranking)
			print inferred_ranking,tau0
			print ''
			result[0].append(tau0)

			print 'Running normal truth inference...'
			score,NTI_ranking,NTI_truth=NTI_main(answer=answer,objects=objects,turn=10)
			tau1=tau_distance(true_ranking,NTI_ranking)
			print NTI_ranking,tau1
			#print 'MAE between truth:',MAE(truth,NTI_truth)
			print ''
			result[1].append(tau1)

			print 'Running fair truth inference...'
			score,FTI_ranking,FTI_truth=FTI_main(answer=answer,objects=objects,turn=10)
			tau2=tau_distance(true_ranking,FTI_ranking)
			print FTI_ranking,tau2
			#print 'MAE between truth:',MAE(truth,FTI_truth)
			print ''
			result[2].append(tau2)

			print [sum(result[0])/(t+1),sum(result[1])/(t+1),sum(result[2])/(t+1)]

			# if tau1-tau2>0.15:
			# 	print truth
			# 	print bias
			# 	print sigma
			# 	print objects
			# 	print truth
			# 	exit()

		for i in range(0,3):
			result[i]=sum(result[i])/20.0

		print '*'*10,result,'*'*10

		f=open('result/synthetic_n%d.txt'%worker_n,'a')
		f.write(str(compare_ratio)+'\t'+str(result)+'\n')
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

def realworld_crime():

	f=open('realworld/crime/truth.txt')
	truth=eval(f.readline())
	f.close()

	all_answer=[]
	label=[]

	fnames=os.listdir('realworld/crime')
	for fname in fnames:
		if 'crime' in fname:
			answer=[]
			f=open('realworld/crime/'+fname)
			for row in f:
				answer.append(eval(row))
				all_answer.append(eval(row))
				label.append(fname.split('.')[0][6:])
			f.close()

			# if len(answer)<3:
			# 	print 'Ignore %s'%fname
			# 	continue

			# print fname

			# esti_truth,bias,sigma,quality=FTI(answer)
			# print 'FTI'
			# print MAE(esti_truth,truth),
			# for i in range(0,len(esti_truth)):
			# 	for j in range(0,len(esti_truth[i])):
			# 		esti_truth[i][j]=int(esti_truth[i][j]>0.5)
			# TPr,TNr,FPr,FNr=confusion_matrix(inferred=esti_truth,truth=truth)
			# precision=TPr/(TPr+FPr)
			# recall=TPr/(TPr+FNr)
			# print precision,recall,2*precision*recall/(precision+recall)
			# print 'Mean bias'
			# print mean_bias(bias)

			# esti_truth,bias,sigma,quality=NTI(answer)
			# print 'NTI'
			# print MAE(esti_truth,truth),
			# for i in range(0,len(esti_truth)):
			# 	for j in range(0,len(esti_truth[i])):
			# 		esti_truth[i][j]=int(esti_truth[i][j]>0.5)
			# TPr,TNr,FPr,FNr=confusion_matrix(inferred=esti_truth,truth=truth)
			# precision=TPr/(TPr+FPr)
			# recall=TPr/(TPr+FNr)
			# print precision,recall,2*precision*recall/(precision+recall)
			# print ''

	avg_bias=None
	avg_quality=[]
	turn=333
	for i in range(0,turn):
		print i
		esti_truth,bias,sigma,quality=FTI(all_answer)
		if avg_bias==None:
			avg_bias=bias
		else:
			avg_bias=matrix_add(avg_bias,bias)
		if len(avg_quality)==0:
			avg_quality=np.array(quality)
		else:
			avg_quality+=np.array(quality)

	avg_quality/=turn

	for i in range(0,len(avg_bias)):
		for j in range(0,len(avg_bias[i])):
			if avg_bias[i][j]!=None:
				avg_bias[i][j]/=turn

	f=open('crime_bias.txt','w')
	for i in range(0,len(avg_bias)):
		f.write(label[i]+'\t')
		for j in range(0,len(avg_bias[i])):
			f.write(str(avg_bias[i][j])+'\t')
		f.write(str(avg_quality[i]*len(avg_quality))+'\n')
	f.close()

	# print 'ALL'
	# esti_truth,bias,sigma,quality=FTI(all_answer)
	# print 'FTI'
	# print 'Overall MAE :',MAE(esti_truth,truth)
	# for i in range(0,len(esti_truth)):
	# 	for j in range(0,len(esti_truth[i])):
	# 		esti_truth[i][j]=int(esti_truth[i][j]>0.5)
	# TPr,TNr,FPr,FNr=confusion_matrix(inferred=esti_truth,truth=truth)
	# precision=TPr/(TPr+FPr)
	# recall=TPr/(TPr+FNr)
	# print precision,recall,2*precision*recall/(precision+recall)
	# print 'Mean bias'
	# print mean_bias(bias)
	# print 'Overall accuracy :',(TPr+TNr)/(TPr+TNr+FPr+FNr)

	# esti_truth,bias,sigma,quality=NTI(all_answer)
	# print 'NTI'
	# print 'Overall MAE :',MAE(esti_truth,truth)
	# for i in range(0,len(esti_truth)):
	# 	for j in range(0,len(esti_truth[i])):
	# 		esti_truth[i][j]=int(esti_truth[i][j]>0.5)
	# TPr,TNr,FPr,FNr=confusion_matrix(inferred=esti_truth,truth=truth)
	# precision=TPr/(TPr+FPr)
	# recall=TPr/(TPr+FNr)
	# print precision,recall,2*precision*recall/(precision+recall)
	# print ''
	# print 'Overall accuracy :',(TPr+TNr)/(TPr+TNr+FPr+FNr)

if __name__=='__main__':
	realworld_crime()
	# Synthetic()