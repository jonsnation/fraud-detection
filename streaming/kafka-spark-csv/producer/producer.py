from kafka import KafkaProducer
import csv
import json
import time
import random

def create_producer():
    for _ in range(5):
        try:
            producer = KafkaProducer(
                bootstrap_servers='kafka:9092',
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Connected to Kafka successfully.")
            return producer
        except Exception as e:
            print(f"Kafka connection failed: {e}")
            time.sleep(5)
    raise Exception("Failed to connect to Kafka after retries")

producer = create_producer()

sampled_rows = []
sample_limit = random.randint(5, 10)
count = 0


# with open('X_subset_features.csv', mode='r') as file:
with open('reduced_features.csv', mode='r') as file:
    reader =  csv.DictReader(file)

    for row in reader:
       count += 1
       if(len(sampled_rows) < sample_limit):
           sampled_rows.append(row)
       else:
           replace_index = random.randint(0, count -1)
           if (replace_index < sample_limit):
               sampled_rows[replace_index] = row


for row in sampled_rows:
    print(f"Sending transaction: {row}")
    try:
        producer.send('transactions', row)
        time.sleep(0.2)
    except Exception as e:
        print(f"Kafka error: {e}")

producer.flush()
time.sleep(2)
