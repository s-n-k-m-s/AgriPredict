from flask import Flask, request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
crop_model = pickle.load(open('crop_model.pkl','rb'))
soil_model = pickle.load(open('soil_model.pkl','rb'))
#creating flask app
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('soil.html')

@app.route('/crop',methods=['POST'])
def fertile():
  pH = float(request.form['pH'])
  EC = float(request.form['EC'])
  OC = float(request.form['OC'])
  OM = float(request.form['OM'])
  N = int(request.form['Nitrogen'])
  P = int(request.form['Phosporus'])
  K = int(request.form['Potassium'])
  Z = int(request.form['Zinc'])
  Fe = int(request.form['Iron'])
  Cu = int(request.form['Copper'])
  Mn = int(request.form['Manganese'])
  Sand = int(request.form['Sand'])
  Silt = int(request.form['Silt'])
  Clay = int(request.form['Clay'])
  cc = int(request.form['CaCo3'])
  cec = int(request.form['CEC'])
  features = [pH, EC, OC, OM, N, P, K, Z, Fe, Cu, Mn, Sand, Silt, Clay, cc, cec]
  single_pred = np.array(features).reshape(1, -1)
  prediction = soil_model.predict(single_pred)
  if prediction == 'Fertile':
    return render_template('crop.html')
  else:
    return render_template('soil.html', result='Soil is not fertile')

@app.route("/predict",methods=['POST'])
def predict():
  N = int(request.form['Nitrogen'])
  P = int(request.form['Phosporus'])
  K = int(request.form['Potassium'])
  temp = float(request.form['Temperature'])
  humidity = float(request.form['Humidity'])
  ph = float(request.form['Ph'])
  rainfall = float(request.form['Rainfall'])
  feature_list = [N, P, K, temp, humidity, ph, rainfall]
  single_pred = np.array(feature_list).reshape(1, -1)
  prediction = crop_model.predict(single_pred)
  if prediction:
    return render_template('crop.html', result='The crop should be {}'.format(prediction))
  else:
    return render_template('crop.html', result='Sorry, we could not determine the best crop to be cultivated with the provided data.')
#python main
if __name__ == "__main__":
  app.run(debug=True)