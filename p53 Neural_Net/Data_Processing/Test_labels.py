#Test_labels
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)


df = pd.read_excel("Final_Data.xlsx", sheetname = 'Test')
label = df['Pathogenicity']

my_Array = np.zeros([1556, 2], dtype = float)

for index, item in enumerate(label):
	if item == 'Benign':
		my_Array[index][0] = 1.0
	elif item == 'Pathogenic':
		my_Array[index][1] = 1.0

print(my_Array)

#np.savetxt('C_Test_labels.txt', my_Array)