# eth_price_prediction

This LSTM prediction model was part of my Master's thesis using pandas, numpy, tensorflow, matplotlib and pylab libraries. 

## Model Description

To investigate whether investor sentiment, investor attention, and market indices can improve the prediction of cryptocurrency prices, three different LSTM models were developed; each of which was fed with a different combination of predictive features. Five data sets were formed from different feature combinations for the LSTM and three different LSTM architectures are used and compared: a simple LSTM, a classical stacked LSTM and a special stacked LSTM, the bidirectional LSTM. 

The aim is to find the best model for predicting the cryptocurrency in question and possibly to make differentiated statements regarding future feature selection.

### Training and Testing 
The data sets are divided into training and test data sets. This is to prevent data leakage so as not to feed the model with the data to be predicted. The training dataset is used to train the LSTM, and the test dataset is used to test the prediction accuracy. Since the data set is not particularly large due to daily aggregation, a ratio of 90 to 10 was chosen. This means that 90% of the data was used to train the LSTM, and 10% of the data was used for testing. 

Since the LSTM can only read NumPy arrays, the pandas data frames were converted to n-dimensional arrays, where n corresponds to the dimensions of the respective feature datasets.

Since there is no generally accepted method for the exact configuration of an LSTM, this must be done by systematic trial and error. In this study, in addition to varying the number of layers (by building both a stacked LSTM and a bidirectional LSTM), the length of the time steps, the size of the training batch, and the number of neurons were varied to find the best possible combination.

### Building the Model
The model was developed as a sequential model in Keras with TensorFlow as the backend.
The simple LSTM consists of an LSTM layer and a dense layer. A layer is a composition of neurons. The dense layer serves as a kind of connector between the layers. The bidirectional model consists of a bidirectional LSTM layer and a dense layer. The stacked LSTM consists of two LSTM layers and one dense layer. As for further parameters, the loss function and the optimization function have to be defined. The loss function measures the quality of the classification by comparing the forecast and the actual value. Since this is not a classification problem but a regression problem, the mean square error was chosen to calculate the average difference between predicted and actual values. Finally, an optimization function must be determined. The optimization function adjusts the weights of the model based on the loss function. The Adam optimizer was chosen as the optimizer because it has established itself as the default optimizer.
The number of iterations was set to 200. This was determined after random testing based on the validation loss curve. For validation, 10% of the training data was used. Since this is a time series problem, the order of observations has to be kept, and accordingly, the shuffle parameter has been set to False.
