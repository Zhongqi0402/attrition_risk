import requests
import json
#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1"
with open('config.json','r') as f:
    config = json.load(f) 

model_path = config['output_model_path']


#Call each API endpoint and store the responses
response1 = requests.get(URL+':8000/prediction?filename=testdata/testdata.csv').content
response2 = requests.get(URL+':8000/scoring').content
response3 = requests.get(URL+':8000/summarystats').content
response4 = requests.get(URL+':8000/diagnostics').content


#write the responses to your workspace
with open(model_path+"/apireturns.txt", 'w') as fp:
    #fp.write(str(response1)+str(response2)+str(response3)+str(response4))
    # fp.writelines([str(response1), str(response2), str(response3), str(response4)])
    fp.write(str(response1)+"\n")
    fp.write(str(response2)+"\n")
    fp.write(str(response3)+"\n")
    fp.write(str(response4)+"\n")
