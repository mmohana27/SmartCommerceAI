from confluent_kafka import Producer
import random
import time
import json  # ✅ For proper JSON encoding

# Kafka Producer Configuration
conf = {
    'bootstrap.servers': 'localhost:9092',  # Kafka broker address
    'client.id': 'transaction-producer'
}

producer = Producer(conf)

def produce_transaction():
    transaction_data = {
        'user_id': random.randint(1, 1000),
        'amount': random.randint(1, 1000),
        'transaction_id': random.randint(10000, 99999),
        'fraud': random.choice([0, 1])  # 0: No fraud, 1: Fraud
    }

    # ✅ Properly encode the message as JSON bytes
    producer.produce(
        topic='transactions_new',  # ✅ Changed topic to avoid old broken messages
        key=str(transaction_data['transaction_id']),
        value=json.dumps(transaction_data).encode('utf-8')
    )

    print(f"Produced transaction: {transaction_data}")
    producer.flush()

# Simulate producing transactions continuously
while True:
    produce_transaction()
    time.sleep(random.uniform(0.5, 2))  # Random delay between transactions
