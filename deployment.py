import os
import json
import subprocess


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], "ingestedfiles.txt") 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl") 
score_path = os.path.join(config['output_model_path'], "latestscore.txt") 

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    subprocess.run(['cp', dataset_csv_path, prod_deployment_path])
    subprocess.run(['cp', model_path, prod_deployment_path])
    subprocess.run(['cp', score_path, prod_deployment_path])
        
        

if __name__ == '__main__':
    store_model_into_pickle()
