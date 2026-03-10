# Databricks notebook source
# MAGIC %md
# MAGIC # Jayd POC — Step 2: LLM Evaluation
# MAGIC Scores unevaluated prompts using ai_query with Llama 3.3 70B.

# COMMAND ----------

from pyspark.sql.functions import col, current_timestamp, from_json, get_json_object, regexp_replace, udf, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import json, re

# COMMAND ----------

# Delete any existing NULL-scored rows so we can reinsert
spark.sql("DELETE FROM main.jayd_poc.silver_evaluated_prompts WHERE overall_score IS NULL")

# COMMAND ----------

# Find unevaluated prompts
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

# Evaluate using ai_query
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
      'Return ONLY a valid JSON object with these exact keys: ',
      'overall_score (int), clarity_score (int), specificity_score (int), context_score (int), structure_score (int), ',
      'improvement_suggestion (string), improved_prompt (string). ',
      'Do NOT wrap in markdown code fences. Return ONLY the JSON object. ',
      'Prompt to evaluate: "', REPLACE(prompt_text, '"', '\\"'), '"'
    )
  ) AS raw_evaluation
FROM unevaluated_prompts
""")

# COMMAND ----------

# UDF to clean and extract JSON from potentially messy LLM output
@udf(StringType())
def clean_json(text):
    if text is None:
        return None
    # Remove markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove first line if it's ```json or ```
        lines = cleaned.split("\n")
        lines = lines[1:]  # skip opening fence
        cleaned = "\n".join(lines)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try to find JSON object
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    # Validate it's parseable
    try:
        json.loads(cleaned)
        return cleaned
    except:
        return None

# Apply cleaning
cleaned_df = evaluated.withColumn("clean_eval", clean_json(col("raw_evaluation")))

# COMMAND ----------

# Parse the cleaned JSON
parsed = cleaned_df.select(
    "prompt_id", "prompt_text", "user_id", "department", "category", "submitted_at", "source",
    get_json_object("clean_eval", "$.overall_score").cast("int").alias("overall_score"),
    get_json_object("clean_eval", "$.clarity_score").cast("int").alias("clarity_score"),
    get_json_object("clean_eval", "$.specificity_score").cast("int").alias("specificity_score"),
    get_json_object("clean_eval", "$.context_score").cast("int").alias("context_score"),
    get_json_object("clean_eval", "$.structure_score").cast("int").alias("structure_score"),
    get_json_object("clean_eval", "$.improvement_suggestion").alias("improvement_suggestion"),
    get_json_object("clean_eval", "$.improved_prompt").alias("improved_prompt"),
).withColumn("evaluated_at", current_timestamp())

# COMMAND ----------

# Write to silver table
parsed.write.mode("append").saveAsTable("main.jayd_poc.silver_evaluated_prompts")

# COMMAND ----------

total = spark.table("main.jayd_poc.silver_evaluated_prompts").count()
scored = spark.sql("SELECT COUNT(*) FROM main.jayd_poc.silver_evaluated_prompts WHERE overall_score IS NOT NULL").collect()[0][0]
print(f"Silver table: {total} total, {scored} with scores.")

# Show sample
display(spark.sql("""
    SELECT prompt_id, overall_score, clarity_score, specificity_score, context_score, structure_score,
           LEFT(improvement_suggestion, 100) as suggestion_preview
    FROM main.jayd_poc.silver_evaluated_prompts
    WHERE overall_score IS NOT NULL
    ORDER BY overall_score DESC
    LIMIT 10
"""))
