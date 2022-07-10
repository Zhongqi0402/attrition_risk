import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv") 
test_csv_path = os.path.join(config['test_data_path'], "testdata.csv")
figure_path = os.path.join(config['output_model_path']) 


##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    data = pd.read_csv(test_csv_path)
    y_hat = model_predictions(test_csv_path)
    y = data['exited'].values.reshape(-1, 1).ravel()
   
    heatmap = confusion_matrix(y, y_hat)
    plot = sns.heatmap(heatmap)
    plot.figure.savefig(figure_path + "/" + "confusionmatrix.png")
    return 





if __name__ == '__main__':
    score_model()
