# Jayd POC — Prompt Intelligence Platform on Databricks

A Databricks-native POC inspired by [jayd.ai](https://jayd.ai) that demonstrates a **Prompt Intelligence Platform**: ingest, score, improve, and execute prompts with LLM-powered analytics.

## Architecture

```
Seed JSON → UC Volume → Autoloader → bronze_prompts
                                          │
                              ai_query(Llama 3.3 70B)
                                          │
                                   silver_evaluated_prompts
                                          │
                                    Streamlit App
                              (4 tabs: Lab, Analytics,
                               Templates, Live Feed)
```

## Components

| Component | Description |
|-----------|-------------|
| `00_setup.py` | Creates schema, volume, tables, uploads seed data |
| `01_autoloader_ingest.py` | Autoloader: JSON → bronze_prompts |
| `02_llm_evaluation.py` | ai_query scores + improves prompts → silver |
| `app.py` | Streamlit app with 4 tabs |

## Quick Start

```bash
# Deploy the bundle
databricks bundle deploy -t prod --profile=fe-sandbox-tko

# Run the pipeline
databricks bundle run jayd_pipeline -t prod --profile=fe-sandbox-tko

# Deploy the app
databricks apps deploy jayd-prompt-intelligence \
  --source-code-path /Workspace/Users/<your-email>/jayd-databricks-poc/src/app \
  --profile=fe-sandbox-tko
```

## Dataset

200 seed prompts across 6 categories (Marketing, Content, Engineering, Data, Support, Strategy) with deliberately varying quality levels from vague one-liners to detailed, well-structured prompts.

## LLM Evaluation Dimensions

- **Clarity** (0-100): How clear and unambiguous is the prompt?
- **Specificity** (0-100): How specific are the requirements?
- **Context** (0-100): How much relevant context is provided?
- **Structure** (0-100): How well-organized is the prompt?
