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



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'], "testdata.csv") 
model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl") 

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    testData = pd.read_csv(test_data_path)
    X=testData.loc[:,['lastmonth_activity','lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y=testData['exited'].values.reshape(-1, 1).ravel()
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
   
    predicted = model.predict(X)
    
    f1score=metrics.f1_score(predicted,y)
    
    with open(config['output_model_path']+"/"+"latestscore.txt", 'w') as fp:
        fp.write(str(f1score))

    return str(f1score)
        
    
if __name__ == '__main__':
    score_model()