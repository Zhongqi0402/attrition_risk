
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv") 
test_data_path = os.path.join(config['test_data_path'], "testdata.csv") 
model_path = os.path.join(config["prod_deployment_path"], "trainedmodel.pkl")
##################Function to get model predictions
def model_predictions(test_data_path):
    #read the deployed model and a test dataset, calculate predictions
    test_data = pd.read_csv(test_data_path)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    X=test_data.loc[:,['lastmonth_activity','lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y_hat = model.predict(X)
 
    return y_hat.tolist()

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(dataset_csv_path)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics)
    final_list = []
  
    final_list.append(list(newdf.mean(axis=0)))
    final_list.append(list(newdf.median(axis=0)))
    final_list.append(list(newdf.std(axis=0)))

    return final_list

def check_missing_data():
    df = pd.read_csv(dataset_csv_path)
    result = list(df.isna().sum() / df.shape[0])
    return result

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    result = []
    
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing=timeit.default_timer() - starttime
    result.append(timing)
    
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing=timeit.default_timer() - starttime
    result.append(timing)
    return result

##################Function to check dependencies
def outdated_packages_list():

    info=subprocess.check_output(['python', '-m', 'pip', 'list', '--outdated'])
  
    with open('package.txt', 'wb') as f:
        f.write(info)
    
    return info
if __name__ == '__main__':
    model_predictions(test_data_path)
    dataframe_summary()
    check_missing_data()
    execution_time()
    outdated_packages_list()