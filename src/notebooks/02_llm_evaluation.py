# Databricks notebook source
# MAGIC %md
# MAGIC # Jayd POC — Step 2: LLM Evaluation
# MAGIC Scores unevaluated prompts using ai_query with Llama 3.3 70B.

# COMMAND ----------

from pyspark.sql.functions import col, current_timestamp, from_json, get_json_object
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# COMMAND ----------

# Find unevaluated prompts
unevaluated = spark.sql("""
    SELECT b.prompt_id, b.prompt_text, b.user_id, b.department, b.category, b.submitted_at, b.source
    FROM main.jayd_poc.bronze_prompts b
    LEFT ANTI JOIN main.jayd_poc.silver_evaluated_prompts s
    ON b.prompt_id = s.prompt_id
""")

count = unevaluated.count()
print(f"Found {count} unevaluated prompts to process.")

if count == 0:
    dbutils.notebook.exit("No new prompts to evaluate.")

# COMMAND ----------

# Evaluate using ai_query — process in SQL for best performance
unevaluated.createOrReplaceTempView("unevaluated_prompts")

evaluated = spark.sql("""
SELECT
  prompt_id,
  prompt_text,
  user_id,
  department,
  category,
  submitted_at,
  source,
  ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    CONCAT(
      'You are a prompt engineering expert. Evaluate the following prompt on a scale of 0-100 for each dimension. ',
      'Return ONLY valid JSON (no markdown, no explanation) with these exact keys: ',
      'overall_score (int), clarity_score (int), specificity_score (int), context_score (int), structure_score (int), ',
      'improvement_suggestion (string, 1-2 sentences), improved_prompt (string, the rewritten better version). ',
      'Prompt to evaluate: "', prompt_text, '"'
    )
  ) AS raw_evaluation
FROM unevaluated_prompts
""")

# COMMAND ----------

# Parse the JSON evaluation results
parsed = evaluated.select(
    "prompt_id", "prompt_text", "user_id", "department", "category", "submitted_at", "source",
    get_json_object("raw_evaluation", "$.overall_score").cast("int").alias("overall_score"),
    get_json_object("raw_evaluation", "$.clarity_score").cast("int").alias("clarity_score"),
    get_json_object("raw_evaluation", "$.specificity_score").cast("int").alias("specificity_score"),
    get_json_object("raw_evaluation", "$.context_score").cast("int").alias("context_score"),
    get_json_object("raw_evaluation", "$.structure_score").cast("int").alias("structure_score"),
    get_json_object("raw_evaluation", "$.improvement_suggestion").alias("improvement_suggestion"),
    get_json_object("raw_evaluation", "$.improved_prompt").alias("improved_prompt"),
).withColumn("evaluated_at", current_timestamp())

# COMMAND ----------

# Write to silver table
parsed.write.mode("append").saveAsTable("main.jayd_poc.silver_evaluated_prompts")

# COMMAND ----------

total = spark.table("main.jayd_poc.silver_evaluated_prompts").count()
print(f"Silver table now has {total} evaluated prompts.")

# Show sample
display(spark.sql("""
    SELECT prompt_id, overall_score, clarity_score, specificity_score, context_score, structure_score,
           LEFT(improvement_suggestion, 100) as suggestion_preview
    FROM main.jayd_poc.silver_evaluated_prompts
    ORDER BY overall_score DESC
    LIMIT 10
"""))
