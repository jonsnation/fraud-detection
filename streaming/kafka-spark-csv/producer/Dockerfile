FROM python:3.9-slim

# Install netcat (openbsd variant works well with slim)
RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY producer.py .
COPY transactions.csv .
COPY X_subset_features.csv .
COPY reduced_features.csv .
COPY wait-for-kafka.sh .

RUN pip install kafka-python
RUN chmod +x wait-for-kafka.sh

CMD ["./wait-for-kafka.sh"]


