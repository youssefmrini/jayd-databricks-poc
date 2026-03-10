# Databricks notebook source
# MAGIC %md
# MAGIC # Jayd POC — Step 0: Setup
# MAGIC Creates schema, volume, tables, and uploads seed data.

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS main.jayd_poc")
spark.sql("CREATE VOLUME IF NOT EXISTS main.jayd_poc.prompt_landing_zone")

# COMMAND ----------

# Upload seed JSON files to the volume
import os, shutil

volume_path = "/Volumes/main/jayd_poc/prompt_landing_zone/incoming"
dbutils.fs.mkdirs(volume_path)

# Copy seed files from bundle's data directory
bundle_data_path = os.path.join(os.environ.get("DATABRICKS_BUNDLE_PROJECT_ROOT", "/Workspace"), "data", "seed_prompts")

# If running from bundle, files are in the workspace
for fname in ["batch_001.json", "batch_002.json", "batch_003.json"]:
    src = os.path.join(bundle_data_path, fname)
    dst = f"{volume_path}/{fname}"
    if os.path.exists(src):
        dbutils.fs.cp(f"file:{src}", dst)
        print(f"Copied {fname} to volume")
    else:
        # Try workspace path
        ws_src = f"file:/Workspace{os.environ.get('DATABRICKS_BUNDLE_WORKSPACE_ROOT_PATH', '')}/data/seed_prompts/{fname}"
        try:
            dbutils.fs.cp(ws_src, dst)
            print(f"Copied {fname} from workspace to volume")
        except Exception as e:
            print(f"Warning: Could not copy {fname}: {e}")

# COMMAND ----------

# Create bronze_prompts table
spark.sql("""
CREATE TABLE IF NOT EXISTS main.jayd_poc.bronze_prompts (
  prompt_id STRING,
  prompt_text STRING,
  user_id STRING,
  department STRING,
  category STRING,
  submitted_at TIMESTAMP,
  source STRING,
  _ingested_at TIMESTAMP
)
USING DELTA
""")

# COMMAND ----------

# Create silver_evaluated_prompts table
spark.sql("""
CREATE TABLE IF NOT EXISTS main.jayd_poc.silver_evaluated_prompts (
  prompt_id STRING,
  prompt_text STRING,
  user_id STRING,
  department STRING,
  category STRING,
  submitted_at TIMESTAMP,
  source STRING,
  overall_score INT,
  clarity_score INT,
  specificity_score INT,
  context_score INT,
  structure_score INT,
  improvement_suggestion STRING,
  improved_prompt STRING,
  evaluated_at TIMESTAMP
)
USING DELTA
""")

# COMMAND ----------

print("Setup complete: schema, volume, and tables created.")
