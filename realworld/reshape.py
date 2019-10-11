sex={0:'male',1:'female'}
race={1:'white',2:'black',3:'hispanic',4:'asian',5:'native'}

f=open('MTURK_RACE_DEMOGRAPHICS.csv','r')
print ord(f.readline()[18])
demographic={}
for row in f:
	row=row.strip().split(',')
	if row[1]<=3:
		row[1]=0
	else:
		row[1]=1
	key=(int(row[3]),int(row[2]))
	if key[0] not in race or key[1] not in sex:
		continue
	key=(race[key[0]],sex[key[1]])
	if key not in demographic:
		demographic[key]=set([])
	demographic[key].add(row[0])
f.close()

#worker demographic loaded

f=open('resultsv_noage.txt')
attrs=f.readline().strip()
if attrs[-1]=='|':
	attrs=attrs[:-1]
attrs=attrs.split('|')

worker_id=[]
answer={}

count=0
count_all=0
for row in f:
	row=row.strip()
	if row[-1]=='|':
		row=row[:-1]
	row=row.split('|')
	
	answer[row[0]]=[]
	for i in range(1,len(row)):
		answer[row[0]].append([])
		for j in range(0,len(row[i])):
			count_all+=1
			if row[i][j]=='-':
				answer[row[0]][-1].append(None)
			else:
				answer[row[0]][-1].append(int(row[i][j]))
				count+=1
	print answer[row[0]]
f.close()

for item in demographic:
	fname='crime_%s_%s.txt'%(item[0],item[1])
	print fname
	for wid in demographic[item]:
		f=open(fname,'a')
		f.write(str(wid)+'\t'+str(answer[wid])+'\n')
		f.close()


# f=open('answer.txt','w')
# f.write(str(answer)+'\n')
# f.close()