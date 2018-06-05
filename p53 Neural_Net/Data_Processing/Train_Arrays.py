import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)

my_Array = np.zeros([4537, 32], dtype = float)

df = pd.read_excel("Final_Data.xlsx", sheetname = 'Train')

# Mutation Location
x = df['Start']
y = df['End']

for index, item in enumerate(x):
	numerized = item/83257441
	my_Array[index][0] = numerized

for index, item in enumerate(y):
	numerized = item/83257441
	my_Array[index][1] = numerized	

# Wt_Codon
b = df['Wt_Codon']
anomalies = ['Intron_01', 'Intron_02', 'Intron_03', 'Intron_04', 'Intron_05', 'Intron_06', 'Intron_07', 'Intron_08', 'Intron_09', 'Intron_10', '3UTR', '5UTR']

for index, item in enumerate(b):
	if item[0] == 'A':
		my_Array[index][2] = 1.0
	elif item[0] == 'T':
		my_Array[index][3] = 1.0
	elif item[0] == 'G':
		my_Array[index][4] = 1.0
	elif item[0] == 'C':
		my_Array[index][5] = 1.0

	if item[1] == 'A':
		my_Array[index][6] = 1.0
	elif item[1] == 'T':
		my_Array[index][7] = 1.0
	elif item[1] == 'G':
		my_Array[index][8] = 1.0
	elif item[1] == 'C':
		my_Array[index][9] = 1.0

	if item[2] == 'A':
		my_Array[index][10] = 1.0
	elif item[2] == 'T':
		my_Array[index][11] = 1.0
	elif item[2] == 'G':
		my_Array[index][12] = 1.0
	elif item[2] == 'C':
		my_Array[index][13] = 1.0

	for anomaly in anomalies:
		if item == anomaly:
			my_Array[index][2:14] = 0.0	
	
#Mutant Codon
m = df['Mutant_Codon']
anom = ['Ins', 'Del', 'None', 'Splice']

for index, item in enumerate(m):
	if item[0] == 'A':
		my_Array[index][14] = 1.0
	elif item[0] == 'T':
		my_Array[index][15] = 1.0
	elif item[0] == 'G':
		my_Array[index][16] = 1.0
	elif item[0] == 'C':
		my_Array[index][17] = 1.0

	if item[1] == 'A':
		my_Array[index][18] = 1.0
	elif item[1] == 'T':
		my_Array[index][19] = 1.0
	elif item[1] == 'G':
		my_Array[index][20] = 1.0
	elif item[1] == 'C':
		my_Array[index][21] = 1.0

	if item[2] == 'A':
		my_Array[index][22] = 1.0
	elif item[2] == 'T':
		my_Array[index][23] = 1.0
	elif item[2] == 'G':
		my_Array[index][24] = 1.0
	elif item[2] == 'C':
		my_Array[index][25] = 1.0

	for a in anom:
		if item == a:
			my_Array[index][14:26] = 0.0


#UTR/INTRON
for index, item in enumerate(b):
	for intron in anomalies[0:10]:
		if item == intron:
			my_Array[index][26] = 1.0
	for utr in anomalies[10:12]:
		if item == utr:		
			my_Array[index][27] = 1.0


#Ins, Del, None, Splice		
for index, item in enumerate(m):
	if item == anom[0]:
		my_Array[index][28] = 1.0
	elif item == anom[1]:
		my_Array[index][29] = 1.0
	elif item == anom[2]:
		my_Array[index][30] = 1.0
	elif item == anom[3]:
		my_Array[index][31] = 1.0
			

np.savetxt('C_Train_data.txt', my_Array)


