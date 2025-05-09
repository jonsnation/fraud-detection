#!/bin/sh
# Wait for Kafka to be available before starting the consumer

host="kafka"
port="9092"

echo "Waiting for Kafka to be available at $host:$port..."

while ! nc -z $host $port; do
  sleep 1
done

echo "Kafka is up - executing command"
exec "$@"
