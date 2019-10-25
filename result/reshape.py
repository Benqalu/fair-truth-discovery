f=open('synthetic_n100.txt')
a=[]
b=[]
c=[]
for row in f:
	eles=row.split('\t')
	ratio=float(eles[0])
	res=eval(eles[2])
	if res[1]<res[2]:
		t=res[2]
		res[2]=res[1]
		res[1]=t
	a.append((ratio,res[0]))
	b.append((ratio,res[1]))
	c.append((ratio,res[2]))
f.close()

for item in a:
	print(item,end='')
print('')

for item in b:
	print(item,end='')
print('')

for item in c:
	print(item,end='')
print('')