# Databricks notebook source
# MAGIC %md
# MAGIC # Jayd POC — Step 1: Autoloader Ingest
# MAGIC Reads JSON files from the landing zone volume and writes to bronze_prompts.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, col
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

# COMMAND ----------

schema = StructType([
    StructField("prompt_id", StringType(), True),
    StructField("prompt_text", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("department", StringType(), True),
    StructField("category", StringType(), True),
    StructField("submitted_at", StringType(), True),
    StructField("source", StringType(), True),
])

# COMMAND ----------

source_path = "/Volumes/main/jayd_poc/prompt_landing_zone/incoming/"
checkpoint_path = "/Volumes/main/jayd_poc/prompt_landing_zone/_checkpoint"

df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", checkpoint_path + "/schema")
    .option("multiLine", "true")  # JSON arrays
    .schema(schema)
    .load(source_path)
    .withColumn("submitted_at", col("submitted_at").cast("timestamp"))
    .withColumn("_ingested_at", current_timestamp())
)

# COMMAND ----------

(
    df.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path + "/data")
    .trigger(availableNow=True)
    .toTable("main.jayd_poc.bronze_prompts")
)

# COMMAND ----------

count = spark.table("main.jayd_poc.bronze_prompts").count()
print(f"Bronze table now has {count} prompts.")
