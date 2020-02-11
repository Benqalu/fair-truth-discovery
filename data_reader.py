
def read_data(A='race'):

	f=open('./crowd_judgement/truth')
	truth=eval(f.readline())
	f.close()

	answer=[]
	workerid=[]

	f=open('./crowd_judgement/data')
	f.readline()
	for row in f:
		answer.append([])

		row=row.strip()
		if row[-1]=='|':
			row=row[:-1]
		row=row.split('|')
		for i in range(1,len(row)):
			item=row[i]
			answer[-1].append([])
			for e in item:
				if e=='-':
					answer[-1][-1].append(None)
				else:
					answer[-1][-1].append(int(e))
		workerid.append(row[0])
	f.close()

	if A=='race':
		for i in range(0,len(answer)):
			answer[i][0]+=answer[i][1]
			answer[i][3]+=answer[i][2]
			del answer[i][7]
			del answer[i][6]
			del answer[i][5]
			del answer[i][4]
			del answer[i][2]
			del answer[i][1]
		truth[0]+=truth[1]
		truth[3]+=truth[2]
		del truth[7]
		del truth[6]
		del truth[5]
		del truth[4]
		del truth[2]
		del truth[1]

	if A=='gender':
		for i in range(0,len(answer)):
			answer[i][1]+=answer[i][2]
			answer[i][0]+=answer[i][3]
			answer[i]=answer[i][:2]
		truth[1]+=truth[2]
		truth[0]+=truth[3]
		truth=truth[:2]

	return answer,truth,workerid