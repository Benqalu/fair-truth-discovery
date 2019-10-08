from math import log
import numpy,sys,numba
from numba import jit

@jit(nopython=True)
def ranking(pmat,max_iter=100000):
	import numpy as np
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

@jit(nopython=True)
def ranking_force(pmat,max_iter=10000):
	n=len(pmat)
	sequence=[0 for i in range(0,n)]
	score=0.0
	for i in range(0,n-1):
		score+=log(pmat[sequence[i]][sequence[i+1]])

	sequence_=sequence[:]
	for _ in xrange(0,max_iter):
		a=int(uniform(0,n))
		b=int(uniform(0,n))
		while a==b:
			a=int(uniform(0,n))
			b=int(uniform(0,n))
		t=sequence_[a]
		sequence_[a]=sequence_[b]
		sequence_[b]=t
		score_=0.0
		for i in range(0,n-1):
			score_+=log(pmat[sequence_[i]][sequence_[i+1]])
		if score_>score:
			score=score_
			sequence=sequence_

	return numpy.array(sequence)