import numpy as np
from matplotlib import pyplot as plt

theta=[]
accuracy=[]
disparity=[[],[]]

f=open('theta_gender.txt','r')
for row in f:
	row=row.split('\t')
	theta.append(float(row[0]))
	accuracy.append(float(row[1]))
	disparity[0].append(eval(row[2])[0])
	disparity[1].append(eval(row[2])[1])
f.close()

plt.plot(theta,accuracy)
plt.show()