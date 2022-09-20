from flask import Flask, render_template, request
import joblib
import numpy as np


model = joblib.load('laptop_price_ph_model')

app = Flask(__name__)
app.debug = True

# A must at all times
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def indexPage():
	
	# The form is here
    return render_template('index.html')

@app.route('/result' , methods=['POST', 'GET'])
def resultPage():
	a = request.form['a']
	b = request.form['b']
	c = request.form['c']

	arr = np.array([[a,b,c]], dtype=float)
	pred = model.predict(arr)
	predInt = int(pred)

	return render_template('result.html', data=predInt)


# @app.route('/predict', methods=['POST'])
# def resultPage():
# 	data1 = request.form['a']	
# 	data2 = request.form['b']
# 	data3 = request.form['c']
# 	data4 = request.form['d']
# 	data5 = request.form['e']
# 	data6 = request.form['f']
# 	data7 = request.form['g']
# 	arr = np.array([[data1, data2, data3, data4, data5, data6, data7]], dtype=float)
# 	pred = model.predict(arr)
# 	predInt = int(pred)
# 	return render_template('after.html', data=predInt)

# print(type(model))

if __name__ == "__main__":
    app.run(debug=True)
