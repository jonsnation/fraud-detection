FROM python:3.9-slim

RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY consumer.py .
COPY wait-for-kafka.sh .

RUN pip install kafka-python && \
    chmod +x wait-for-kafka.sh

ENTRYPOINT ["./wait-for-kafka.sh"]
CMD ["python", "-u", "consumer.py"]