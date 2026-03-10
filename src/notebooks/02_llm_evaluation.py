# Databricks notebook source
# MAGIC %md
# MAGIC # Jayd POC — Step 2: LLM Evaluation
# MAGIC Scores unevaluated prompts using ai_query with Llama 3.3 70B.

# COMMAND ----------

from pyspark.sql.functions import col, current_timestamp, regexp_replace, get_json_object
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# COMMAND ----------

# Find unevaluated prompts (either not in silver, or in silver with NULL scores)
unevaluated = spark.sql("""
    SELECT b.prompt_id, b.prompt_text, b.user_id, b.department, b.category, b.submitted_at, b.source
    FROM main.jayd_poc.bronze_prompts b
    LEFT ANTI JOIN (
        SELECT prompt_id FROM main.jayd_poc.silver_evaluated_prompts WHERE overall_score IS NOT NULL
    ) s
    ON b.prompt_id = s.prompt_id
""")

count = unevaluated.count()
print(f"Found {count} unevaluated prompts to process.")

if count == 0:
    dbutils.notebook.exit("No new prompts to evaluate.")

# COMMAND ----------

# Delete any existing NULL-scored rows so we can reinsert
spark.sql("""
    DELETE FROM main.jayd_poc.silver_evaluated_prompts
    WHERE overall_score IS NULL
""")

# COMMAND ----------

# Evaluate using ai_query with structured output (returnType)
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
      'You are a prompt engineering expert. Evaluate the following prompt on a scale of 0-100 for each dimension.',
      ' Return a JSON object with keys: overall_score, clarity_score, specificity_score, context_score, structure_score (all integers 0-100),',
      ' improvement_suggestion (1-2 sentence string), improved_prompt (rewritten better version string).',
      ' Prompt to evaluate: "', REPLACE(prompt_text, '"', '\\"'), '"'
    ),
    returnType => 'STRUCT<overall_score: INT, clarity_score: INT, specificity_score: INT, context_score: INT, structure_score: INT, improvement_suggestion: STRING, improved_prompt: STRING>'
  ) AS evaluation
FROM unevaluated_prompts
""")

# COMMAND ----------

# Extract struct fields into columns
parsed = evaluated.select(
    "prompt_id", "prompt_text", "user_id", "department", "category", "submitted_at", "source",
    col("evaluation.overall_score").alias("overall_score"),
    col("evaluation.clarity_score").alias("clarity_score"),
    col("evaluation.specificity_score").alias("specificity_score"),
    col("evaluation.context_score").alias("context_score"),
    col("evaluation.structure_score").alias("structure_score"),
    col("evaluation.improvement_suggestion").alias("improvement_suggestion"),
    col("evaluation.improved_prompt").alias("improved_prompt"),
).withColumn("evaluated_at", current_timestamp())

# COMMAND ----------

# Write to silver table
parsed.write.mode("append").saveAsTable("main.jayd_poc.silver_evaluated_prompts")

# COMMAND ----------

total = spark.table("main.jayd_poc.silver_evaluated_prompts").count()
scored = spark.sql("SELECT COUNT(*) FROM main.jayd_poc.silver_evaluated_prompts WHERE overall_score IS NOT NULL").collect()[0][0]
print(f"Silver table: {total} total, {scored} scored.")

# Show sample
display(spark.sql("""
    SELECT prompt_id, overall_score, clarity_score, specificity_score, context_score, structure_score,
           LEFT(improvement_suggestion, 100) as suggestion_preview
    FROM main.jayd_poc.silver_evaluated_prompts
    WHERE overall_score IS NOT NULL
    ORDER BY overall_score DESC
    LIMIT 10
"""))
