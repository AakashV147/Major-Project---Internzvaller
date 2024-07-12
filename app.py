from flask import Flask, render_template, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    features = [float(x) for x in request.form.values()]

    # Make prediction using the loaded model
    prediction = model.predict([features])[0]

    # Prepare response to send back to the user
    if prediction == 1:
        result = 'Heart Disease Detected'
    else:
        result = 'No Heart Disease Detected'

    return render_template('result.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5001)

