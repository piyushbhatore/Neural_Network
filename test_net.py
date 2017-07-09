import pandas
import numpy as np
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


train_data['bias'] = 1					#adding the bias column to feature vector
#print(train_data)
hiddendim = 7						#no. of dimnesion in hidden layers
inputdim = 15						#no. of dimension in input layer
outputdim = 1						#no. of dimension in oupput layer
#print(train_data)
#initialising the temp and weight matrixs

#decrease the number of zeroes to balance the data




sup = np.vectorize(np.log)
hiddenweights = 2*np.random.rand(inputdim,hiddendim)-1
outputweights = 2*np.random.rand(hiddendim+1,outputdim)-1
t = train_data


mat = np.loadtxt('weights.txt')
mat = np.matrix(mat)
print(mat)
print(mat.shape)
hiddenweights = np.reshape(mat[0,0:inputdim*hiddendim],(inputdim,hiddendim))
outputweights = np.reshape(mat[0,inputdim*hiddendim:],(1+hiddendim,outputdim))

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
test_data["bias"] = 1
test_data = np.matrix(test_data)
temp = test_data.dot(hiddenweights)
temp = sigmoid(temp)
temp = np.append(temp,np.ones((test_data.shape[0],1)),axis=1)
temp = temp.dot(outputweights)
temp = sigmoid(temp)
outputfile = open("predictions.csv","w")
outputfile.write("id,salary\n")
for i in range(temp.shape[0]):
	temp[i] = (temp[i]>0.5)
	outputfile.write(str(id_list[i])+","+str(int(temp[i,0]))+"\n")
print(np.sum(temp))
