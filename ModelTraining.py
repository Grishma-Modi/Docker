# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Dataset Link: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
import pandas as pd
import numpy as np


df=pd.read_csv('BankNote_Authentication.csv')

df.head()


# Independent and Dependent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X.head()

y.head(20)


# Train Test Split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

# Prediction
y_pred=classifier.predict(X_test)


# Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)


score



# Create a Pickle file using serialization 
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


classifier.predict([[2,3,4,1]])


