import numpy as np

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