import csv
import numpy as np
import pandas as pd

data1 =  np.loadtxt("mySubmission_NeuNet_Layer_4.csv",skiprows = 1,delimiter = ",")
data2 =  np.loadtxt("mySubmission_NeuNet_Layer_1.csv",skiprows = 1,delimiter = ",")
data3 =  np.loadtxt("mySubmission_GraBoo.csv",skiprows = 1,delimiter = ",")

cnt12 = 0;
cnt13 = 0;
cnt23 = 0;

for i in range(0,884262):
	sum12 = 0;
	sum13 = 0;
	sum23 = 0;

	for j in range(0,40):
		sum12 += (data1[i][j] - data2[i][j]) * (data1[i][j] - data2[i][j])
		sum23 += (data2[i][j] - data3[i][j]) * (data2[i][j] - data3[i][j])
		sum13 += (data1[i][j] - data3[i][j]) * (data1[i][j] - data3[i][j])

	if (sum12 < sum23 and sum12 < sum13):
		cnt12 += 1;
		for j in range(0,40):
			data1[i][j] = data1[i][j] * 0.5 + data2[i][j] * 0.5

	if (sum23 < sum12 and sum23 < sum13):
		cnt23 += 1;
		for j in range(0,40):
			data1[i][j] = data2[i][j] * 0.5 + data3[i][j] * 0.5

	if (sum13 < sum12 and sum13 < sum23):
		cnt13 += 1;
		for j in range(0,40):
			data1[i][j] = data1[i][j] * 0.5 + data3[i][j] * 0.5

	if (i % 100000 == 0):
		print(i);

df = pd.DataFrame(data1)
del df[0]

df.to_csv("final_Submission.csv")