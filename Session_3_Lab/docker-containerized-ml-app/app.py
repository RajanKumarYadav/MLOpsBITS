from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
# Initialize Flask application
app = Flask(__name__)

knn_model = joblib.load('knn_model.joblib') # This can be your Model Registry/or any cloud location

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json    

    # Dataframe Creation
    df = pd.DataFrame(data["data"])
    input = df

    predictions = knn_model.predict(input)
    final_predictions = pd.DataFrame(list(predictions),columns = ["Flower Species"]).to_dict(orient="records")
 
    return jsonify(final_predictions) 


if __name__ == '__main__':
    app.run(debug=True,host = "0.0.0.0", port = 5000)