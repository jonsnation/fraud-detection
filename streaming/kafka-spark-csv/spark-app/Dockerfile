FROM bitnami/spark:latest

RUN pip install requests
RUN pip install pandas

USER root

COPY spark_app.py /opt/spark_app.py

ENV PYTHONUNBUFFERED=1

