#========================================================================
# author 					: vishal
#========================================================================

#
import flask
import json
import requests
import random
import os
import os.path

from sklearn import metrics
import numpy as np
import pandas as pd
import cPickle as pickle


#
server 			= flask.Flask(__name__)
# server_port 	= 8000


#========================================================================
# utils
#========================================================================

#
def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

#
def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)


def get_prediction_html(table_json,lab):
	print len(table_json)
	html = '<!doctype html><html><body><p> Percent = '+str(lab)+'</p>'
	table = '<table style="text-align:center; text-transform:uppercase;"><tr><th>name</th><th>precision</th><th>recall</th><th>f1-score</th><th>support</th></tr>'
	
	for i in range(len(table_json) - 1):
		print table_json[i]
		try:
			table+='<tr><td>' + str(table_json[i]['title']) + '</td><td>'+ str(table_json[i]['values'][0])+'</td><td>'+ str(table_json[i]['values'][1])+'</td><td>'+ str(table_json[i]['values'][2])+'</td><td>'+ str(table_json[i]['values'][3])+'</td>'
		except:
			None
	html+= table + '</table></body></html>'
	return html


# performs prediction
def get_prediction(csv):
	#
	colname	=	['Name','Info']
	mylist 	= 	[]
	file 	= 	csv

	#
	for chunk in  pd.read_csv(file, names=colname,chunksize=20000):
		mylist.append(chunk)

	#
	dataset = 	pd.concat(mylist, axis= 0)
	del mylist

	#
	df 		=	pd.DataFrame(dataset)
	df 		= 	df.dropna(axis=0,how='any')

 	#
	array 	= 	df.values
	X  		= 	array[:,1]  
	y 		= 	array[:,0] 

	#
	x_test 	= 	X
	y_test 	= 	y

	#
	predicted_svm = loaded_model.predict(x_test)
	np.mean(predicted_svm == y_test)
	
	#
	result = parse_ml_output_to_json(str(metrics.classification_report(y_test, predicted_svm, target_names=y_test)))
	lab 			= str(np.mean(predicted_svm == y_test))

	#
	return get_prediction_html(result,lab)
	


#========================================================================
# endpoints
#========================================================================

# serves a prediction to the client
@server.route("/get-prediction", methods=['POST'])
def serve_prediction():
	#
	if flask.request.method == 'POST':
		print (flask.request.form)
		print (flask.request.form.to_dict())
		file = flask.request.files['csv']
	else:
		print("no file")

    #
	return get_prediction(file)


@server.route("/test", methods=['POST'])
def text_prediction():
	k1= "'" + str(flask.request.form['label1']) +"'"
	vis=[k1]
	predicted_svm1 = loaded_model.predict(vis)
	return '<!doctype html><html><body>' + str(predicted_svm1) + '</body></html>'

# serves the webpage to the client
@server.route("/")
def init():
	content = get_file('index.html')
	return flask.Response(content, mimetype="text/html")

def parse_ml_output_to_json(sample):
	#
	sample = sample.replace("\n\n", "\n")
	sample_parts = sample.split("\n")

	parsed_output = {}
	parsed_output['title'] = [''].extend(sample[0])
	
	count = 0
	#
	for part in sample_parts[1:]:
		# part.replace(" ","")

		part = list(part.split(" "))
		part=[x for x in part if x]
		# for ndx in range(len(part)):
		# 	if part[ndx] in ["", '']:
		# 		del part[ndx]
		print (part)
		title 	= (" ").join(part[0:len(part)-4])
		values 	= part[len(part)-4:len(part)]

		output_dict = {}
		output_dict['title'] = title
		output_dict['values']  = values

		parsed_output[count] = output_dict

		count+=1	

	return parsed_output



if __name__ == '__main__':
	filename = 'finalized_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	server.run(host='0.0.0.0', port=8000, debug=True)

#========================================================================
# starts the server
# server.run(port=server_port)