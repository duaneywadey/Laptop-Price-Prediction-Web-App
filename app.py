from flask import Flask, render_template, request
import joblib
import numpy as np


model = joblib.load('updated_new_laptop_price_ph_model')

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



if __name__ == "__main__":
    app.run(debug=True)
