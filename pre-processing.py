from truth_inference import *


def pre(gtruth, bias, answer):

    def disparityd(t):
        p0 = 0
        p1 = 0
        for i in range(len(t[0])):
            if t[0][i] < 0.5:
                p0 += 1
        for i in range(len(t[1])):
            if t[1][i] < 0.5:
                p1 += 1
        return p0, p1, (p1) / (len(t[1])) - (p0) / (len(t[0]))

    def acc(gtruth, truth):
        correctm = 0
        wrongm = 0
        correctf = 0
        wrongf = 0
        for i in range(len(gtruth[0])):
            if gtruth[0][i] == 1 and truth[0][i] > 0.5:
                correctm += 1
            elif gtruth[0][i] == 0 and truth[0][i] <= 0.5:
                correctm += 1
            else:
                wrongm += 1
        for i in range(len(gtruth[1])):
            if gtruth[1][i] == 1 and truth[1][i] > 0.5:
                correctf += 1
            elif gtruth[1][i] == 0 and truth[1][i] <= 0.5:
                correctf += 1
            else:
                wrongf += 1
        accuracym = correctm / (correctm + wrongm)
        accuracyf = correctf / (correctf + wrongf)
        accuracy = (correctm + correctf) / (correctm + correctf + wrongm + wrongf)
        return accuracym, accuracyf, accuracy

    blist = sorted(bias.items(), key=lambda x: x[1], reverse=False)

    def reject(m):
        answerv = []
        wlist = []
        for i in range(m):
            wlist.append(int(blist[i][0]))
        for i in range(len(answer)):
            if i not in wlist:
                answerv.append(answer[i])
        truth1, sigma, quality = NTI(answerv, VLDB=False)

        d1, d2, disparity = disparityd(truth1)
        a1, a2, accuracy = acc(gtruth, truth1)
        return (disparity, accuracy)

    bsort = []
    for i in blist:
        bsort.append(i[1])
    # threshold values
    tlist = []
    # disparity
    dlist = []
    # accuracy
    alist = []
    for i in range(1, 100):
        threshold = bsort[i - 1]
        d, a = reject(i)
        tlist.append(threshold)
        dlist.append(d)
        alist.append(a)

    return tlist, dlist, alist