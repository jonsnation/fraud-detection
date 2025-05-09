from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import requests

# Create SparkSession
spark = SparkSession.builder \
    .appName("KafkaSparkConsumer") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define schema
'''schema = StructType([
    StructField("TransactionAmt", StringType()),
    StructField("TransactionDT", StringType()),
    StructField("card1", StringType()),
    StructField("card4_freq", StringType()),
    StructField("card6_freq", StringType()),
    StructField("addr1", StringType()),
    StructField("dist1", StringType()),
    StructField("P_emaildomain_freq", StringType()),
    StructField("R_emaildomain_freq", StringType()),
    StructField("M1_freq", StringType()),
    StructField("M4_freq", StringType()),
    StructField("M5_freq", StringType()),
    StructField("M6_freq", StringType()),
    StructField("M9_freq", StringType()),
    StructField("C1", StringType()),
    StructField("C2", StringType()),
    StructField("C8", StringType()),
    StructField("C11", StringType()),
    StructField("V18", StringType()),
    StructField("V21", StringType()),
    StructField("V97", StringType()),
    StructField("V133", StringType()),
    StructField("V189", StringType()),
    StructField("V200", StringType()),
    StructField("V258", StringType()),
    StructField("V282", StringType()),
    StructField("V294", StringType()),
    StructField("V312", StringType()),
    StructField("DeviceType_freq", StringType()),
    StructField("id_15_freq", StringType()),
    StructField("id_28_freq", StringType()),
    StructField("id_29_freq", StringType()),
    StructField("id_31_freq", StringType()),
    StructField("id_35_freq", StringType()),
    StructField("id_36_freq", StringType()),
    StructField("id_37_freq", StringType()),
    StructField("id_38_freq", StringType())
]) '''

schema = StructType([
    StructField("V233", StringType()),
    StructField("V132", StringType()),
    StructField("C5", StringType()),
    StructField("V261", StringType()),
    StructField("V134", StringType()),
    StructField("V302", StringType()),
    StructField("DeviceInfo", StringType()),
    StructField("V184", StringType()),
    StructField("V183", StringType()),
    StructField("V239", StringType()),
    StructField("V224", StringType()),
    StructField("V291", StringType()),
    StructField("id_31", StringType()),
    StructField("V228", StringType()),
    StructField("V319", StringType()),
    StructField("V185", StringType()),
    StructField("id_16", StringType()),
    StructField("V235", StringType()),
    StructField("V258", StringType()),
    StructField("id_01", StringType()),
    StructField("V213", StringType()),
    StructField("V9", StringType()),
    StructField("V98", StringType()),
    StructField("V320", StringType()),
    StructField("V232", StringType()),
    StructField("id_02", StringType()),
    StructField("V204", StringType()),
    StructField("V115", StringType()),
    StructField("V172", StringType()),
    StructField("V252", StringType()),
    StructField("V276", StringType()),
    StructField("V210", StringType()),
    StructField("V180", StringType()),
    StructField("id_36", StringType()),
    StructField("V46", StringType()),
    StructField("V51", StringType()),
    StructField("V103", StringType()),
    StructField("TransactionDT", StringType()),
    StructField("TransactionAmt", StringType()),
    StructField("ProductCD", StringType()),
    StructField("card4", StringType()),
    StructField("C8", StringType()),
    StructField("C9", StringType()),
    StructField("card3", StringType()),
    StructField("C6", StringType()),
    StructField("card1", StringType()),
    StructField("addr1", StringType()),
    StructField("addr2", StringType()),
    StructField("P_emaildomain", StringType()),
    StructField("DeviceType", StringType()),
    StructField("id_17", StringType()),
    StructField("id_28", StringType())
])

# Read from Kafka
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "transactions") \
    .option("startingOffsets", "earliest") \
    .load()

# Parse JSON from Kafka
df_json = df_raw.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

# Cast all fields to DoubleType
for field in schema.fieldNames():
    df_json = df_json.withColumn(field, col(field).cast(DoubleType()))

# Define foreachBatch function to POST to API
def send_to_api(batch_df, batch_id):
    import json
    import pandas as pd

    # Convert to pandas
    records = batch_df.toPandas().to_dict(orient='records')

    for i, record in enumerate(records):
        try:
            #response = requests.post("http://host.docker.internal:5000/predict_stream", json=record)
            response = requests.post("http://host.docker.internal:5000/predict_gnn_with_context", json=record)

            result = response.json()

            print(f"[{i}] Model {result.get('model_used')}: {result.get('prediction')}, Fraud Probability: {result.get('fraud_probability')}")
            #print(f"[{i}] Sent. Status: {response.status_code} Response: {response.json()}")
            #print(f"[{i}] Sent. Prediction: {result['prediction']} | Fraud Probability: {result['fraud_probability']} | Autoencoder Flagged: {result['autoencoder_flagged']}")

        except Exception as e:
            print(f"[{i}] Error sending to API: {e}")

# Stream and send
query = df_json.writeStream \
    .foreachBatch(send_to_api) \
    .outputMode("append") \
    .start()

query.awaitTermination()
