import argparse, joblib
from os import path

from train import load_data

from utils import load_models, check_inputs

from flask import Flask, request, jsonify
app = Flask(__name__)

# Load models
model, tf = load_models()  


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check inputs
        x = check_inputs(request.json['features'])
        y_hat = model.predict(tf.transform(x))
    
        return jsonify(output={"y_hat": y_hat.tolist()}, status=200, message='Model Working')


@app.route('/predict_test')
def predict_test():
    X, y = load_data()
    X_tf = tf.transform(X)
    y_hat = model.predict(X_tf)

    return 'Predict'

    

@app.route('/')
def hello_world():
    return 'Hello!'




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Iris classifier api 0.0.1')
#     parser.add_argument('--host', default='localhost', type=str)
#     parser.add_argument('--port', default=5000, type=str)
#     parser.add_argument('--debug', default=True, type=str)
#     args = vars(parser.parse_args()) 

#     # Load vars
#     model, tf = load_models()    

    # app.run(port=args['port'], debug=args['debug'])



