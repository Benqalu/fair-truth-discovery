def FTI(answer):

	n=len(answer)
	m=len(answer[0])
	r=[len(answer[0][j]) for j in range(0,m)]

	#Step 0: Give random initial numbers
	truth=[[uniform(0,1) for k in range(0,r[j])] for j in range(0,m)]
	bias=[[uniform(-1.0,1.0) for j in range(0,m)] for i in range(0,n)]
	sigma=[uniform(0,0.2) for i in range(0,n)]
	quality=[1.0/n for i in range(0,n)]
	last_truth=np.array([])

	for _ in range(0,30):

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

		Step 3: Calculate worker quality
		Option 1
		quality=np.array(sigma)**2
		quality/=quality.sum()

		#Option 2
		# sum_quality=1E-10
		# for i in range(0,n):
		# 	Ns=0
		# 	up=1E-10
		# 	down=1E-10
		# 	for j in range(0,m):
		# 		for k in range(0,r[j]):
		# 			if answer[i][j][k]!=None:
		# 				Ns+=1
		# 				down+=(answer[i][j][k]-truth[j][k]-bias[i][j])**2
		# 	up+=chi2.ppf(q=0.95,df=Ns)
		# 	quality[i]=up/down

		# sum_quality=sum(quality)
		# for i in range(0,n):
		# 	quality[i]/=sum_quality

	return truth,bias,sigma,quality