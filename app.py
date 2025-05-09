from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Dashboard landing page

@app.route('/run_recommender')
def run_recommender():
    os.system('python modules/recommender.py')
    return 'Recommender module executed.'

@app.route('/run_forecasting')
def run_forecasting():
    os.system('python modules/demand_forecasting.py')
    return 'Demand forecasting executed.'

@app.route('/run_image_classifier')
def run_classifier():
    os.system('python modules/classifier.py')
    return 'Image classification executed.'

@app.route('/run_fraud_detection')
def run_fraud_detection():
    os.system('start cmd /k "python modules/fraud_detection_consumer.py"')
    return 'Fraud detection consumer started in new window.'

@app.route('/run_transaction_producer')
def run_transaction_producer():
    os.system('start cmd /k "python modules/transaction_producer.py"')
    return 'Transaction producer started in new window.'

if __name__ == '__main__':
    app.run(debug=True)
