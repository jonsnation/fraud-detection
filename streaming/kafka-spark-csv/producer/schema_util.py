import csv
from pyspark.sql.types import StructType, StructField, StringType

csv_file = "producer/reduced_features.csv"  # adjust path if needed

with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    headers = next(reader)  # first row = header

schema = StructType([StructField(col, StringType(), True) for col in headers])

print(schema.simpleString())
print("\n")

# Optional: print it in Spark code format
for col in headers:
    print(f'.add("{col}", StringType()) \\')
