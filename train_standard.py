import csv
import numpy as np
import pandas
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

train_data = pandas.read_csv('train.csv')		#reading data using pandas from train.csv
train_data = train_data.drop('id', 1)			#droppping unnecessary cloumns
def sig(x):						#defining the sigmoid function
    return 1 / (1 + np.exp(-x))
sigmoid = np.vectorize(sig)
p = train_data.loc[train_data['salary']==1]
q = train_data.loc[train_data['salary']==0]
q = q[:int(p.shape[0]*1.3)]
train_data = p.append(q)

Y = train_data['salary']				#creating the Y matrix
train_data = train_data.drop('salary',1)		#removing the salary from the X matrix
accuracy = 7000
redgelambda = 1
epsilon = 0.000005
np.random.seed(1)
#handling the absent data
numberlist = {}
m = {}
s = {}

intcolumn = [key for key in dict(train_data.dtypes) if dict(train_data.dtypes)[key] in ['int64']]
for iter in intcolumn:
	if iter == "education-num":
		continue
	train_data[[iter]] = train_data[[iter]].astype(float)
	m[iter] = train_data[iter].mean()
	s[iter] = train_data[iter].std()
	train_data[iter] = (train_data[iter] - train_data[iter].mean())/train_data[iter].std()

objectcolumn = [key for key in dict(train_data.dtypes) if dict(train_data.dtypes)[key] in ['object']]
for iter in objectcolumn:				#data featuring
	train_data[iter].replace(' ?',np.NaN,inplace=True)				#replacing the ? with Nan
	train_data[iter].replace(np.NaN,train_data[iter].mode()[0],inplace=True)	#replacing again with th mode
	numberlist[iter] = dict(enumerate(train_data[iter].unique()))				#making the list for converting features into numbers
	numberlist[iter] = dict(zip(numberlist[iter].values(), numberlist[iter].keys()))
	train_data[iter] = train_data[iter].map(numberlist[iter])
t = train_data
train_data = np.matrix(train_data)

test_data = pandas.read_csv('kaggle_test_data.csv')
id_list = test_data["id"]
test_data = test_data.drop('id',1)
#print(test_data)
for iter in intcolumn:
	if iter == "education-num":
		continue
	if iter == "salary":
		continue
	test_data[[iter]] = test_data[[iter]].astype(float)
	test_data[iter] = (test_data[iter] - m[iter])/s[iter]

for iter in objectcolumn:				#data featuring
	numberlist[iter][' ?'] = t[iter].mode()[0]
	#test_data[iter].replace(' ?',np.NaN,inplace=True)				#replacing the ? with Nan
	#test_data[iter].replace(np.NaN,t[iter].mode()[0],inplace=True)	#replacing again with th mode
	test_data[iter] = test_data[iter].map(numberlist[iter])
#print(test_data)

test_data = np.matrix(test_data)

clf = LogisticRegression()
clf.fit(train_data, Y)
p1 = clf.predict(test_data)

print(np.sum(p1))

gnb = GaussianNB()
gnb.fit(train_data, Y)
p2 = gnb.predict(test_data)

print(np.sum(p2))

clf = svm.SVC()
clf.fit(train_data,Y)
p3 = clf.predict(test_data)

print(np.sum(p3))
print(p1.shape)
print(id_list.shape)
outputfile = open("predictions_1.csv","w")
outputfile.write("id,salary\n")
for i in range(p1.shape[0]):
	p1[i] = (p1[i]>0.5)
	outputfile.write(str(id_list[i])+","+str(int(p1[i]))+"\n")

outputfile = open("predictions_2.csv","w")
outputfile.write("id,salary\n")
for i in range(p2.shape[0]):
	p2[i] = (p2[i]>0.5)
	outputfile.write(str(id_list[i])+","+str(int(p2[i]))+"\n")

outputfile = open("predictions_3.csv","w")
outputfile.write("id,salary\n")
for i in range(p3.shape[0]):
	p3[i] = (p3[i]>0.5)
	outputfile.write(str(id_list[i])+","+str(int(p3[i]))+"\n")