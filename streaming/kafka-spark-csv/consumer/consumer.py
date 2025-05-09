from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='kafka:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='transaction-consumer-group',
    auto_offset_reset='earliest',  # Only applies on first run
    enable_auto_commit=True
)

print("Consumer is running and waiting for messages...", flush=True)

for message in consumer:
    print(f"Received: {message.value}", flush=True)

