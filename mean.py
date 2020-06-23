import os 

f = "./imm3943/IMM-Frontal Face DB SMALL/asf/"
fileList = os.listdir(f)
fileList.sort()
for i in fileList:
	for line in open((f+"/"+i),"r"):
		if line.startswith("#"):
			continue
		else:
			print (line)
	break 




