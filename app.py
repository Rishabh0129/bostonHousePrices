import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
# load the model
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # input will be captured in json
    data = request.json['data']
    print(np.array(list(data.values())).reshape(1, -1))
    newdata = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(newdata)
    print(output[0])
    return jsonify(output[0])
@app.route('/pred',methods=['POST'])
def pred():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    out=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(out))

if __name__ == "__main__":
    app.run(debug=True)
