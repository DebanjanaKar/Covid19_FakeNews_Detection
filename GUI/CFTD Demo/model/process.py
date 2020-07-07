from flask import Flask, render_template, request, jsonify
import predict_tweet

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('form.html')

@app.route('/process', methods=['POST'])
def process():

	name = request.form['name']

	if name :#and email:
		try:

			newName = predict_tweet.predict(name)
			return jsonify({'name' : newName})
		
		except:

			newName = "Wrong Tweet URL / ID"
			return jsonify({'name' : newName})
		

	return jsonify({'error' : 'Missing data!'})

if __name__ == '__main__':
	app.run(debug=True)