from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, check_missing_data, execution_time, outdated_packages_list
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('filename')
    y_hat = model_predictions(filename)
    return "Predictions: " + str(y_hat)+'\n'


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    score = score_model()
    return "score: " + score + "\n"

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    stats = dataframe_summary()
    return "Mean, Median, Std of all numerical columns: " + str(stats) + "\n"

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def other_diagnostics():        
    #check timing and percent NA values
    missing_data_percent = check_missing_data()
    time = execution_time()
    package = outdated_packages_list()
    data_return = "missing data percentage of each column: " + str(missing_data_percent) + "\n"
    time_return = "Time to run ingestion and training step: " + str(time) + "\n"
    package_return = "All Packages info: " + str(package) + "\n"
    return '{}{}{}'.format(data_return, time_return, package_return)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)