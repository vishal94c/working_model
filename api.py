import flask
from flask import Flask, render_template, request
from sklearn import metrics
import numpy as np
import pandas as pd
import cPickle as pickle
import json


app = Flask(__name__)
@app.route("/")
@app.route("/index")
def index():
	return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':
		colname=['Name','Info']
		mylist = []

		file = request.files['csv']
		for chunk in  pd.read_csv(file, names=colname,chunksize=20000):
			mylist.append(chunk)

		dataset = pd.concat(mylist, axis= 0)
		del mylist

		df=pd.DataFrame(dataset)
		df=df.dropna(axis=0,how='any')


		array = df.values
		X = array[:,1]  #data
		y = array[:,0]  #label

		x_test = X
		y_test = y

		predicted_svm = loaded_model.predict(x_test)
		np.mean(predicted_svm == y_test)
		
		rest={}
		rest['table'] = str(metrics.classification_report(y_test, predicted_svm,target_names=y_test))

		lab=str(np.mean(predicted_svm == y_test))

		return render_template('index.html', label=lab)

@app.route('/test', methods=['POST'])
def make_prediction_test():
	if request.method=='POST':
		X=request.data('label1')
		y=request.data('data1')
		x_test = X
		y_test = y

		predicted_svm = loaded_model.predict(x_test)
		lab1=str(np.mean(predicted_svm == y_test))
		return render_template('index.html', label11=lab1)

if __name__ == '__main__':
	filename = 'finalized_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	app.run(host='0.0.0.0', port=8000, debug=True)



