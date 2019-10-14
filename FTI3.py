--- FTI.py	(original)
+++ FTI.py	(refactored)
@@ -1,4 +1,4 @@
-from __future__ import division
+
 import numpy as np
 from numpy.random import *
 import sys,itertools,os
@@ -36,7 +36,7 @@
 	n=pmat.shape[0]
 	wins=np.sum(pmat,axis=0)
 	params=np.ones(n,dtype=float)
-	for _ in xrange(max_iter):
+	for _ in range(max_iter):
 		tiled=np.tile(params,(n,1))
 		combined=1.0/(tiled+tiled.T)
 		np.fill_diagonal(combined,0)
@@ -50,7 +50,7 @@
 
 
 def tau_distance(A,B):
-	pairs=itertools.combinations(range(0,len(A)),2)
+	pairs=itertools.combinations(list(range(0,len(A))),2)
 	distance=0
 	for x,y in pairs:
 		a=A[x]-A[y]
@@ -236,7 +236,7 @@
 	for j in range(0,category_n):
 		answer[i].append([])
 		for k in range(0,len(truth[j])):
-				l=range(0,worker_n)
+				l=list(range(0,worker_n))
 				shuffle(l)
 				for i in l:
 					if uniform(0,1)<portion[i] or count[j][k]==0:
@@ -490,7 +490,7 @@
 		else:
 			total_truth=matrix_add(total_truth,truth)
 
-		print _+1,'/',turn,'\r',
+		print(_+1,'/',turn,'\r', end=' ')
 		sys.stdout.flush()
 
 	f=open('result/truth_FTI.txt','w')
@@ -567,7 +567,7 @@
 		else:
 			total_truth=matrix_add(total_truth,truth)
 
-		print _+1,'/',turn,'\r',
+		print(_+1,'/',turn,'\r', end=' ')
 		sys.stdout.flush()
 
 	f=open('result/truth_NTI.txt','w')
@@ -603,59 +603,59 @@
 	compare_ratio=1.0
 	turn=100
 
-	print 'Number of objects:',object_n
-	print 'Number of workers:',worker_n
-	print 'Number of categories:',category_n
-
-	for c in reversed(range(1,11)):
+	print('Number of objects:',object_n)
+	print('Number of workers:',worker_n)
+	print('Number of categories:',category_n)
+
+	for c in reversed(list(range(1,11))):
 
 		compare_ratio=0.1*c
 
-		print 'compare_ratio =',compare_ratio
+		print('compare_ratio =',compare_ratio)
 
 		result=[[],[],[]]
 
 		for t in range(0,turn):
 
-			print ''
-			print '#'*30
-			print ''
-
-			print 'turn =',t
-
-			print 'Compare ratio',compare_ratio
+			print('')
+			print('#'*30)
+			print('')
+
+			print('turn =',t)
+
+			print('Compare ratio',compare_ratio)
 
 			answer,truth,objects,true_ranking,bias,sigma=data_generator(object_n,worker_n,category_n,compare_ratio)
-			print 'Data generated.\n'
-
-			print 'True ranking:'
-			print true_ranking
-			print ''
+			print('Data generated.\n')
+
+			print('True ranking:')
+			print(true_ranking)
+			print('')
 
 			score,inferred_ranking=infer_ranking(truth=truth,objects=objects)
-			print 'Inferred ranking:'
+			print('Inferred ranking:')
 			tau0=tau_distance(true_ranking,inferred_ranking)
-			print inferred_ranking,tau0
-			print ''
+			print(inferred_ranking,tau0)
+			print('')
 			result[0].append(tau0)
 
-			print 'Running normal truth inference...'
+			print('Running normal truth inference...')
 			score,NTI_ranking,NTI_truth=NTI_main(answer=answer,objects=objects,turn=1)
 			tau1=tau_distance(true_ranking,NTI_ranking)
-			print NTI_ranking,tau1
+			print(NTI_ranking,tau1)
 			#print 'MAE between truth:',MAE(truth,NTI_truth)
-			print ''
+			print('')
 			result[1].append(tau1)
 
-			print 'Running fair truth inference...'
+			print('Running fair truth inference...')
 			score,FTI_ranking,FTI_truth=FTI_main(answer=answer,objects=objects,turn=1)
 			tau2=tau_distance(true_ranking,FTI_ranking)
-			print FTI_ranking,tau2
+			print(FTI_ranking,tau2)
 			#print 'MAE between truth:',MAE(truth,FTI_truth)
-			print ''
+			print('')
 			result[2].append(tau2)
 
-			print [sum(result[0])/(t+1),sum(result[1])/(t+1),sum(result[2])/(t+1)]
+			print([sum(result[0])/(t+1),sum(result[1])/(t+1),sum(result[2])/(t+1)])
 
 			# if tau1-tau2>0.15:
 			# 	print truth
@@ -668,7 +668,7 @@
 		for i in range(0,3):
 			result[i]=sum(result[i])/turn
 
-		print '*'*10,result,'*'*10
+		print('*'*10,result,'*'*10)
 
 		f=open('result/synthetic_n%d.txt'%worker_n,'a')
 		f.write(str(compare_ratio)+'\t'+str(result)+'\n')
@@ -747,14 +747,14 @@
 	delete_count=0
 	while True:
 
-		print 'Delete :',delete_count
+		print('Delete :',delete_count)
 
 		avg_truth=None
 		avg_bias=None
 		avg_quality=[]
 		turn=1
 		for i in range(0,turn):
-			print i,'/',turn,'\r',
+			print(i,'/',turn,'\r', end=' ')
 			sys.stdout.flush()
 			esti_truth,bias,sigma,quality=FTI(all_answer)
 			if avg_bias==None:
@@ -781,11 +781,11 @@
 				if avg_truth[i][j]!=None:
 					avg_truth[i][j]/=turn
 
-		print 'MAE :',MAE(avg_truth,truth)
+		print('MAE :',MAE(avg_truth,truth))
 
 		TPr,TNr,FPr,FNr=confusion_matrix(inferred=avg_truth,truth=truth)
-		print TPr,TNr,FPr,FNr
-		print 'Accuracy :',(TPr+TNr)/(TPr+TNr+FPr+FNr),'\t','Precision :',TPr/(TPr+FPr),'\t','Recall :',TPr/(TPr+FNr)
+		print(TPr,TNr,FPr,FNr)
+		print('Accuracy :',(TPr+TNr)/(TPr+TNr+FPr+FNr),'\t','Precision :',TPr/(TPr+FPr),'\t','Recall :',TPr/(TPr+FNr))
 
 		ii=-1
 		jj=-1
@@ -877,12 +877,12 @@
 		if row[-1]==',':
 			row=row[:-1]
 		row=row.split(',')
-		for i in xrange(1,len(row)):
+		for i in range(1,len(row)):
 			row[i]=float(row[i])-0.5
 		stastic_bias[worker_id]=row[1:]
 	f.close()
 
-	print stastic_bias
+	print(stastic_bias)
 
 
 
