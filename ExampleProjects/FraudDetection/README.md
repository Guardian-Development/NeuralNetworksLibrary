# Fraud Detection Neural Network Example
The aim of this project is to show how the Neural Networks library can be integrated into a small, simple web application. 

## Starting the Application 
1. Download and unzip the Credit Card Transaction Data Set from: https://www.kaggle.com/dalpozz/creditcardfraud 

2. Update the AppSettings.json file to have the correct location of the downloaded file
3. Run the web project : dotnet run

## How To Use 
1. Navigate to the Train Tab. 
2. From here you can enter the amount of epochs to train for. (1000 produces accurate results)
3. Once trained, go to the Predict Tab. 
4. Click predict to get predictions for all data rows not used in training. You will also be presented with a report of how successful your trained Neural Network was.

## Adapting
- To change the Learning rate, Momentum, Minimum Error Threshold and Thread Count when using the Neural Network edit the appsettings.json file. 

## Notes
This is a small application designed to show how a Neural Network can solve quite a complex problem. It has not been made with extensibility in mind and does not support more than one user.

