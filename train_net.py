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
train_data = np.matrix(train_data)
hiddenvalues = np.zeros((train_data.shape[0],hiddendim))
outputvalues = np.zeros((train_data.shape[0],1))
Y = np.matrix(Y).transpose()
for i in range(accuracy):
	#Forward Propogation

	hiddenvalues = train_data.dot(hiddenweights)
	hiddenvalues = sigmoid(hiddenvalues)
	hiddenvalues = np.append(hiddenvalues,np.ones((train_data.shape[0],1)),axis=1)
	outputvalues = hiddenvalues.dot(outputweights)
	outputvalues = sigmoid(outputvalues)
	#print(np.append(outputvalues,Y,axis=1))
	#Backward Propagation
	delta3 = outputvalues - Y
	delta2 = np.multiply(np.multiply(delta3.dot(outputweights.transpose()),hiddenvalues),(1-hiddenvalues))
	dW1 = np.dot(train_data.transpose(),delta2)
	dW2 = hiddenvalues.transpose().dot(delta3)
	dW1[-1,:] = np.sum(delta2, axis=0)
	dW2[-1,:] = np.sum(delta3, axis=0)
	#Redeg

	dW2 += redgelambda * outputweights
	dW1[:,0:hiddendim] += redgelambda * hiddenweights

	#Grad
	hiddenweights += -epsilon * dW1[:,0:hiddendim]
	outputweights += -epsilon * dW2
	print(np.sum(np.multiply(Y,(sup(outputvalues)))+np.multiply((1-Y),(sup(1-outputvalues)))))
m1 = np.reshape(hiddenweights,(1,inputdim*hiddendim))
m2 = np.reshape(outputweights,(1,(hiddendim+1)*outputdim))
mat = np.append(m1,m2)
np.savetxt('weights.txt',mat)