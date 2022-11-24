from flask import Flask, render_template, request
import joblib
import numpy as np


model = joblib.load('new_labeled_laptop_price_ph_model')

app = Flask(__name__)
app.debug = True

# A must at all times
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def indexPage():
	
	# The form is here
    return render_template('index.html')

# Acer	Asus	Dell	HP	Huawei	Lenovo

@app.route('/result' , methods=['POST', 'GET'])
def resultPage():
	a = int(request.form['a'])
	b = request.form['b']
	c = request.form['c']

	if a == 1:
		arr = np.array([[1,0,0,0,0,0,b,c]], dtype=float)
		pred = model.predict(arr)
		predInt = int(pred)

	elif a == 2:
		arr = np.array([[0,1,0,0,0,0,b,c]], dtype=float)
		pred = model.predict(arr)
		predInt = int(pred)

	elif a == 3:
		arr = np.array([[0,0,1,0,0,0,b,c]], dtype=float)
		pred = model.predict(arr)
		predInt = int(pred)

	elif a == 4:
		arr = np.array([[0,0,0,1,0,0,b,c]], dtype=float)
		pred = model.predict(arr)
		predInt = int(pred)

	elif a == 5:
		arr = np.array([[0,0,0,0,1,0,b,c]], dtype=float)
		pred = model.predict(arr)
		predInt = int(pred)

	elif a == 6:
		arr = np.array([[0,0,0,0,0,1,b,c]], dtype=float)
		pred = model.predict(arr)
		predInt = int(pred)

	elif a == 7:
		arr = np.array([[0,0,0,0,0,0,b,c]], dtype=float)
		pred = model.predict(arr)
		predInt = int(pred)


	return render_template('result.html', data=predInt)



if __name__ == "__main__":
    app.run(debug=True)
