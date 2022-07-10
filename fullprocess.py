import training
import scoring
import deployment
import diagnostics
import reporting
import logging
from ingestion import merge_multiple_dataframe
from training import train_model
from deployment import store_model_into_pickle
from reporting import score_model
import pickle
import json
import os
import pandas as pd
from sklearn import metrics
logging.basicConfig(level = logging.INFO)

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

prod_folder = config['prod_deployment_path']
source_data_folder = config['input_folder_path']
data_path = config['output_folder_path']

def full_automation():
    
    logging.info("Checking if new data present and need to ingest")
    old_data = []
    with open(prod_folder + "/ingestedfiles.txt", 'r') as file:
        for line in file:
            old_data.append(line.strip())
            
    source_files = os.listdir(source_data_folder)
    
    if set(old_data) != set(source_files):
        merge_multiple_dataframe()
        


    logging.info("checking for model drift")
    logging.info("Predicting using old model")
    with open(prod_folder + "/trainedmodel.pkl", 'rb') as file:
        old_mopdel = pickle.load(file)
    data = pd.read_csv(data_path + "/finaldata.csv")
    X=data.loc[:,['lastmonth_activity','lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y=data['exited'].values.reshape(-1, 1).ravel()
    predicted = old_mopdel.predict(X)
    f1score=metrics.f1_score(predicted,y)
    
    logging.info("Compare old score and new score and check for model drift")
    with open(prod_folder + "/latestscore.txt", 'rb') as file:
        old_score = float(file.readline().strip())
    if old_score < f1score:
        logging.info("MODEL DRIFT: start training new model")
        train_model()
        store_model_into_pickle()
        score_model()
        os.system("python diagnostics.py")
        os.system("python apicalls.py")
    else:
        logging.info("Did not detect model drift")
            



if __name__ == '__main__':
    full_automation()



