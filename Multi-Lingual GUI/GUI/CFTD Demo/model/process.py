from flask import Flask, render_template, request, jsonify,Markup
import predict_multi_lingual

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('form.html')

@app.route('/process', methods=['POST'])
def process():

	name = request.form['name']

	if name :#and email:
		try:

			newName = predict_multi_lingual.predict(name)
			newName = Markup(newName)
			return jsonify({'name' : newName})
		
		except:

			newName = "Wrong Tweet URL / ID"
			return jsonify({'name' : newName})
		

	return jsonify({'error' : 'Missing data!'})

if __name__ == '__main__':
	app.run(debug=True)