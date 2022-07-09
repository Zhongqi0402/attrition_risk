from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging


logging.basicConfig(level = logging.INFO)

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv") 
model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl") 


#################Function for training the model
def train_model():
    
    logging.info("reading in data")
    data = pd.read_csv(dataset_csv_path)
    
    logging.info("training logistic regression")
    X=data.loc[:,['lastmonth_activity','lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y=data['exited'].values.reshape(-1, 1).ravel()
    print(X.shape)
    print(y.shape)
    #use this logistic regression for training
    lr_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    lr_model.fit(X, y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info("Saving trained model")
    pickle.dump(lr_model, open(model_path, 'wb'))
    

if __name__ == '__main__':
    train_model()