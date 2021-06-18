#Stokes, Jeff
#CS 5710
#Final Assignment - Part 2

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#suppress numpy-related warnings
import warnings
warnings.filterwarnings('ignore')

#Import training set
dataset_train = pd.read_csv('data_train.csv')
training_set = dataset_train.iloc[:,1:2].values
# print(training_set[0:9,:])

# print()
#Feature scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
# print(training_set_scaled[0:9,:])

#Create data structure with 60 timesteps & 1 output
X_train = []
y_train = []
for i in range(60, 4000):
	X_train.append(training_set_scaled[i-60:i,0])
	y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# print(X_train[0:1,:])
# print()
# print(np.shape(y_train))

#Reshape array
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#import keras libraries & packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# #Initialize RNN
regressor = Sequential()

#Add first LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Add second LSTM layer and Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Add third LSTM layer and Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Add fourth LSTM layer and Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Add output layer
regressor.add(Dense(units=1))

#Compile the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fit RNN to training set
regressor.fit(X_train, y_train, epochs=200, batch_size=32)

#Making Predictions & visualising results

#Get test data
dataset_test = pd.read_csv('data_test.csv')
test_data = dataset_test.iloc[:,1:2].values

print(test_data[0:9,:])

#Get predicted data
dataset_total = pd.concat((dataset_train['Value'],dataset_test['Value']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(60,1060):
	X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_output = regressor.predict(X_test)
predicted_output = sc.inverse_transform(predicted_output)

print(predicted_output[0:9,:])

#Visualizing results
plt.plot(test_data, color='red', label='Test Data Outputs')
plt.plot(predicted_output, color='blue', label = 'Predicted Outputs for Test Data')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show(block=True)