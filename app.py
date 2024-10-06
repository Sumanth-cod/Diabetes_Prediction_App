import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model (SVM Classifier) and the scaler
model = pickle.load(open("classification-model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define the home page route
@app.route("/")
def home():
    return render_template("index.html")

# Define the predict route for form submission
@app.route("/predict", methods=["POST"])
def predict():
    # Extract form data
    int_features = [float(x) for x in request.form.values()]  # Collects user input and converts it to float
    features = [np.array(int_features)]  # Convert input data to a numpy array

    # Standardize the input data using the same scaler used during model training
    scaled_features = scaler.transform(features)

    # Use the model to make predictions
    prediction = model.predict(scaled_features)

    # Determine the prediction result
    if prediction[0] == 0:
        result = "No Diabetes"
    else:
        result = "Diabetes"

    # Render the template and display the result
    return render_template("index.html", prediction_text=f"The Person is {result}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
