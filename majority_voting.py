from random import uniform

def majority_voting(answer):
	truth=[]
	for j in range(len(answer[0])):
		truth.append([])
		for k in range(len(answer[0][j])):
			truth[-1].append([0,0])
	for i in range(0,len(answer)):
		for j in range(0,len(answer[i])):
			for k in range(0,len(answer[i][j])):
				if answer[i][j][k]!=None:
					truth[j][k][answer[i][j][k]]+=1
	for j in range(0,len(truth)):
		for k in range(0,len(truth[j])):
			if truth[j][k][0]>truth[j][k][1]:
				truth[j][k]=0
			elif truth[j][k][0]<truth[j][k][1]:
				truth[j][k]=1
			elif truth[j][k][0]==truth[j][k][1]:
				truth[j][k]=int(uniform(0,1)>=0.5)
			else:
				truth[j][k]=None
	return truth