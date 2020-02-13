from truth_inference import *
from massaging import *

def FairTD_Post_MV(answer,theta):
	truth=MV(answer)
	truth=massaging(truth=truth,theta=theta,cat=2)
	return truth

def FairTD_Post_EM(answer,theta):
	truth,_,_,_=NTI(answer,CATD=False)
	truth=massaging(truth=truth,theta=theta,cat=2)
	return truth

def FairTD_Post_CATD(answer,theta):
	truth,_,_,_=NTI(answer,CATD=True)
	truth=massaging(truth=truth,theta=theta,cat=2)
	return truth
