#%%
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#%%
# loading the dataset to a Pandas Dataframe
credit_card_data = pd.read_csv('/Users/sexybeast69/credit_cardf/creditcard.csv')

#%%
# frist 5 rows of the dataset 
credit_card_data.head()
                               
                               
# %%
# last 5 rows of the dataset 
credit_card_data.tail()
# %%

# dataset infromations
credit_card_data.info()
# %%
# checking the number of missing values in each column
credit_card_data.isnull().sum()
# %%
# distribution of legit transactions of fradualent transactions
credit_card_data['Class'].value_counts()
# %%
# the dataset is very unbalanced  0 -- > Normal transaction  1 --> fradulent transaction
# seperating the data forn anlaysis 
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)
# %%
# statistical measures of the data 
legit.Amount.describe()
# %%
fraud.Amount.describe()

#%%
# compare the means for all the Vs of both transactions to show the difference 
credit_card_data.groupby('Class').mean()
# %%
#Under-Sampling 
#Build a sample dataset containing similar distribution of normal transaction and Fraudulent Transaction
#Number of Fraudulent Transactions --> 492
legit_sample = legit.sample(n=492)

# %%
#Concatenating two Dataframes
new_dataset = pd.concat([legit_sample, fraud], axis = 0)
# %%
new_dataset.head()
# %%
new_dataset.tail()
# %%
new_dataset['Class'].value_counts()
# %%
new_dataset.groupby('Class').mean()
# %%
# splitting the data into Features & Targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print(X)
# %%
print(Y)
# %%
#Splitting the data into Training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state = 2)
# 80% train, 20% test
# %%
print(X.shape, X_train.shape, X_test.shape)
# %%
#Model training : Logistic Regression for binary 
model = LogisticRegression()
# %%
#training the logistic Regression Model with Training Data
model.fit(X_train, Y_train)
# %%
# Model Evaluation : Accuracy Score
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data: ' , training_data_accuracy)
# %%
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Testing data: ' , test_data_accuracy)
# %%
