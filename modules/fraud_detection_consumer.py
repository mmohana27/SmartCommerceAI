from confluent_kafka import Consumer, KafkaException
import json
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Kafka Consumer Configuration
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fraud-detection-consumer',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(conf)

def consume_transactions():
    consumer.subscribe(['transactions_new'])
    try:
        while True:
            msg = consumer.poll(1.0)  # Timeout after 1 second
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            
            # Parse transaction data
            transaction = json.loads(msg.value().decode('utf-8'))
            print(f"Consumed transaction: {transaction}")
            
            # Extract features (we'll use simple features for demo purposes)
            features = np.array([[transaction['amount'], transaction['user_id']]])
            
            # Simple fraud detection model (RandomForestClassifier for demo)
            model = RandomForestClassifier()
            model.fit([[500, 1], [1000, 2], [200, 3], [50, 4]], [0, 1, 0, 0])  # Dummy training data
            
            fraud_prediction = model.predict(features)
            print(f"Fraud detected: {fraud_prediction[0]}")
    
    finally:
        consumer.close()

# Start consuming transactions
consume_transactions()
