from flask import Flask, render_template, request
from flask_cors import cross_origin
from pyngrok import ngrok
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder="/content/drive/MyDrive/Rain-Prediction-main/template")

# Load your machine learning model
model = pickle.load(open("/content/drive/MyDrive/Rain-Prediction-main/models/cat.pkl", "rb"))
print("Model Loaded")

# Set your ngrok auth token
ngrok.set_auth_token("2lJrYy0qRZseHOUQJ0rq6awCMbi_4wC9WWAWnaERTSVTGNXLi")

# Connect ngrok to your Flask app
public_url = ngrok.connect(addr="5000")
print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000/\"")

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index1.html")

@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        try:
            # DATE
            date = request.form['date']
            day = float(pd.to_datetime(date, format="%Y-%m-%d").day)
            month = float(pd.to_datetime(date, format="%Y-%m-%d").month)

            # MinTemp
            minTemp = request.form.get('mintemp', None)
            if minTemp is None or minTemp == '':
                return "Error: 'mintemp' field is required."
            minTemp = float(minTemp)

            # MaxTemp
            maxTemp = request.form.get('maxtemp', None)
            if maxTemp is None or maxTemp == '':
                return "Error: 'maxtemp' field is required."
            maxTemp = float(maxTemp)

            # Rainfall
            rainfall = request.form.get('rainfall', None)
            if rainfall is None or rainfall == '':
                return "Error: 'rainfall' field is required."
            rainfall = float(rainfall)
            evaporation = float(request.form.get('evaporation', 0))
            sunshine = float(request.form.get('sunshine', 0))
            windGustSpeed = float(request.form.get('windgustspeed', 0))
            windSpeed9am = float(request.form.get('windspeed9am', 0))
            windSpeed3pm = float(request.form.get('windspeed3pm', 0))
            humidity9am = float(request.form.get('humidity9am', 0))
            humidity3pm = float(request.form.get('humidity3pm', 0))
            pressure9am = float(request.form.get('pressure9am', 0))
            pressure3pm = float(request.form.get('pressure3pm', 0))
            temp9am = float(request.form.get('temp9am', 0))
            temp3pm = float(request.form.get('temp3pm', 0))
            cloud9am = float(request.form.get('cloud9am', 0))
            cloud3pm = float(request.form.get('cloud3pm', 0))
            location = float(request.form.get('location', 0))
            winddDir9am = float(request.form.get('winddir9am', 0))
            winddDir3pm = float(request.form.get('winddir3pm', 0))
            windGustDir = float(request.form.get('windgustdir', 0))
            rainToday = float(request.form.get('raintoday', 0))

            input_lst = [location, minTemp, maxTemp, rainfall, evaporation, sunshine,
                         windGustDir, windGustSpeed, winddDir9am, winddDir3pm, windSpeed9am, windSpeed3pm,
                         humidity9am, humidity3pm, pressure9am, pressure3pm, cloud9am, cloud3pm, temp9am, temp3pm,
                         rainToday, month, day]

            # Convert input list to numpy array
            input_array = np.array(input_lst).reshape(1, -1)

            # Predict the result using the loaded model
            pred = model.predict(input_array)[0]

            # Return result
            prediction_text = "Rainy" if pred == 1 else "Sunny"
            return render_template("predictor1.html", prediction=prediction_text)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template("predictor1.html")

if __name__ == '__main__':
    app.run(port=5000, debug=False)