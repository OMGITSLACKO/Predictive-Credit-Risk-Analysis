from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

# Load the pre-trained model (ensure this file exists in your project directory)
model = joblib.load("gradient_boosting_winning_model.pkl")

app = Flask(__name__)

# Index route with basic instructions
@app.route('/')
def index():
    return (
        "<h1>Welcome to the Credit Risk Predictor API!</h1>"
        "<p>You can use the following endpoints:</p>"
        "<ul>"
        "<li><strong>/predict</strong> - Use a JSON POST request to get a prediction.</li>"
        "<li><strong>/simulate</strong> - Use a user-friendly form to enter features and get a prediction.</li>"
        "</ul>"
    )

# REST API endpoint: expects JSON input.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if "features" not in data:
            return jsonify({"error": "Missing key 'features' in JSON payload"}), 400

        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Simulated environment: a webpage with a dark theme and golden accent.
@app.route('/simulate', methods=['GET', 'POST'])
def simulate():
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Risk Predictor - Simulator</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #1e1e1e;
                color: #e0e0e0;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 600px;
                margin: 50px auto;
                background: #2e2e2e;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
                border-left: 5px solid #d4af37; /* Golden accent */
                border-radius: 5px;
            }
            h1, h2 {
                text-align: center;
            }
            form {
                display: flex;
                flex-direction: column;
            }
            label {
                margin-top: 10px;
            }
            input[type="text"] {
                padding: 10px;
                font-size: 16px;
                margin-top: 5px;
                background: #3e3e3e;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 4px;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #d4af37;
            }
            input[type="submit"] {
                margin-top: 20px;
                padding: 10px;
                font-size: 18px;
                background: #d4af37;
                color: #1e1e1e;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background: #c69e35;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background: #3e3e3e;
                border-left: 5px solid #d4af37;
                color: #e0e0e0;
            }
            a {
                display: block;
                text-align: center;
                margin-top: 20px;
                text-decoration: none;
                color: #d4af37;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Credit Risk Predictor</h1>
            <p>Enter the 13 feature values below:</p>
            <form method="post">
                {% for i in range(1, 14) %}
                    <label for="f{{i}}">Feature {{ i }}:</label>
                    <input type="text" id="f{{i}}" name="f{{i}}" required>
                {% endfor %}
                <input type="submit" value="Predict">
            </form>
            {% if prediction is defined %}
                <div class="result">
                    <h2>Prediction: {{ prediction_text }}</h2>
                    <p>(0 = Default, 1 = Non-default)</p>
                </div>
            {% endif %}
            <a href="/">Back to Home</a>
        </div>
    </body>
    </html>
    """

    if request.method == "POST":
        try:
            features = [float(request.form.get(f"f{i}")) for i in range(1, 14)]
            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)
            prediction_text = "Non-default (1)" if int(prediction[0]) == 1 else "Default (0)"
            return render_template_string(html_template, prediction=True, prediction_text=prediction_text)
        except Exception as e:
            return render_template_string(html_template, prediction=True, prediction_text=f"Error: {str(e)}")
    else:
        return render_template_string(html_template)

if __name__ == '__main__':
    app.run(debug=True)
