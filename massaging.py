import matplotlib.pyplot as plt
import math
from random import *
from copy import deepcopy

def massaging(theta,truth,r=1,cat=2):

    th=theta

    def point(x,y):
        list = []
        for i in range(th):
            list.append((x[i],y[i]))
        return list

    def disparityd(t):
        p0=0
        p1=0
        for i in range(len(t[0])):
            if t[0][i] < 0.5:
                p0+=1
        for i in range(len(t[1])):
            if t[1][i] < 0.5:
                p1+=1
        return p0, p1, (p1) / (len(t[1])) - (p0) / (len(t[0]))

    def disparityi(t):
        p0=0
        p1=0
        for i in range(len(t[0])):
            if t[0][i] == 0:
                p0+=1
        for i in range(len(t[1])):
            if t[1][i] == 0:
                p1+=1
        return p0, p1, (p1) / (len(t[1])) - (p0) / (len(t[0]))

    def disparitys(t,c):
        p0 = 0
        p1 = 0
        for i in range(cat):
            if i == c:
                for j in range(len(t[i])):
                    if t[i][j] < 0.5:
                        p0 += 1
            else:
                for j in range(len(t[i])):
                    if t[i][j] < 0.5:
                        p1 += 1

        return p0, p1, (p0) / (len(t[0])) - (p1) / (len(t[1]))

    def dis(m,p0,p1,len0,len1):
        return (p1 - m) / len1 - (p0 + m) / len0

    def mdis(dis,p0,p1,len0,len1):
        return (len0*p1 - len1*p0 - len0*len1*dis) / (len0 + len1)

    dlist = []
    for c in range(cat):
        p0,p1,d = disparitys(truth,c)
        dlist.append(d)
    p = dlist.index(min(dlist))

    pr = []
    de = []
    for i in range(cat):
        if i == p:
            for j in range(len(truth[i])):
                if truth[i][j] >= 0.5:
                    pr.append(j)
        else:
            for j in range(len(truth[i])):
                if truth[i][j] < 0.5:
                    de.append(j)

    disD = abs(dlist[p])
    len0 = len(truth[p])
    len1 = 0
    for i in range(cat):
        if i != p:
            len1 += len(truth[i])
    M = (disD * len0 * len1) / (len0 + len1)
    M = math.ceil(M)
    p0,p1,d = disparitys(truth,p)

    mlist = []
    thetas = [0.01*i for i in range(th,th+1)]
    for t in thetas:
        mlist.append(max(0,math.ceil(mdis(t,p0,p1,len0,len1))))

    newtruth = [[], []]
    for i in range(len(truth)):
        for j in truth[i]:
            if j >= 0.5:
                newtruth[i].append(1)
            else:
                newtruth[i].append(0)

    def mass(m):
        lenp=len(pr)
        lend=len(de)
        plist=[i for i in range(lenp)]
        shuffle(plist)
        dlist=[i for i in range(lend)]
        shuffle(dlist)
        newt=deepcopy(newtruth)
        count=0
        for i in plist:
            if newt[p][i] == 1:
                if count == m:
                    break
                else:
                    newt[p][i]=0
                    count+=1
        count=0
        for i in dlist:
            if newt[abs(1 - p)][i] == 0:
                if count == m:
                    break
                else:
                    newt[abs(1 - p)][i]=1
                    count+=1

        return newt

    displist = [[] for i in range(th)]
    acclist=[[] for i in range(th)]

    newt = mass(mlist[0])

    return newt