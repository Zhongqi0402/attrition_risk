import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging


logging.basicConfig(level = logging.INFO)

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    logging.info("Reading source data and append into a single dataframe")
    filenames = os.listdir(input_folder_path)
    final_df = pd.DataFrame()
    source_list = []
    for file in filenames:
        path = os.getcwd() + '/' + input_folder_path + '/' + file
        source_list.append(input_folder_path + '/' + file)
        current_df = pd.read_csv(path)
        final_df = final_df.append(current_df)
    
    logging.info("drop duplicates in the final dataframe")
    final_df = final_df.drop_duplicates()
    
    logging.info("Saving final data to csv file")
    final_df.to_csv(output_folder_path + '/finaldata.csv', index=False)
    
    logging.info("Writing source data path to file and save")
    with open(output_folder_path + '/' + "ingestedfiles.txt", 'w') as fp:
        for file in source_list:
            fp.write(file+'\n')
    
    
    


if __name__ == '__main__':
    merge_multiple_dataframe()
