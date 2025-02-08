from flask import Flask, request, jsonify, render_template
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Load and train the model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])
    output = iris.target_names[prediction[0]]
    return render_template('index.html', prediction_text=f'Predicted Iris class: {output}')

if __name__ == "__main__":
    app.run(debug=True)