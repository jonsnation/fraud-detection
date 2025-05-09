#!/bin/bash

echo "Waiting for Kafka to be ready..."
# Wait until Kafka broker is ready
while ! nc -z kafka 9092; do
  sleep 2
done

echo "Kafka is ready. Starting producer..."
echo "Now running producer.py"
python producer.py
