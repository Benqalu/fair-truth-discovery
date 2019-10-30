def confusion_matrix(esti_truth,truth,bound=0.5):
	TP=0
	TN=0
	FP=0
	FN=0
	total=0
	for i in range(0,len(esti_truth)):
		for j in range(0,len(esti_truth[i])):
			if esti_truth[i][j]==None or truth[i][j]==None:
				continue
			total+=1
			if esti_truth[i][j]>bound and truth[i][j]>bound:
				TP+=1
			if esti_truth[i][j]<bound and truth[i][j]<bound:
				TN+=1
			if esti_truth[i][j]>bound and truth[i][j]<bound:
				FP+=1
			if esti_truth[i][j]<bound and truth[i][j]>bound:
				FN+=1

	return TP,TN,FP,FN