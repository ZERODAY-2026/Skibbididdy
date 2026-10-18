# DataMind — Talk to Your Data

> **ZeroDay OSS Hacktoberfest 2026 — Track PS1: DataMind**  
> An open-source AI data assistant that lets you query your data warehouse in plain English using **Google Gemma** and **Snowflake**.

---

## Overview

DataMind translates natural language questions into dialect-correct Snowflake SQL queries, executes them live on your Snowflake warehouse, and presents verifiable tabular results with audit metadata (including Snowflake Query ID and latency).

### How it works:
1. **Schema Context**: Discovers accessible tables from Snowflake.
2. **SQL Generation**: Google Gemma writes a targeted, read-only SQL query.
3. **Safety Verification**: Blocks non-`SELECT` statements to guarantee read-only execution.
4. **Live Execution**: Runs query against Snowflake, displaying the results alongside the Snowflake Query ID for full auditability.

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- Snowflake account
- Access to Google Gemma (via AWS Bedrock Mantle, Ollama, or an OpenAI-compatible gateway)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ZERODAY-2026/Skibbididdy.git
cd Skibbididdy

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and fill in your connection details:
```bash
cp .env.example .env
```

```ini
# Snowflake Connection
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=DEMO_DB
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=PUBLIC

# Google Gemma LLM Settings
LLM_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/openai/v1
LLM_API_KEY=your_api_key
LLM_MODEL=google.gemma-4-31b
```

### 4. Usage
Run the CLI assistant with a question:
```bash
python main.py "How many records are in the orders table?"
```

Or run interactively:
```bash
python main.py
```

---

## Roadmap

- [x] Initial Snowflake connector and query runner
- [x] Google Gemma SQL generation integration
- [x] Read-only query validation
- [ ] Automated schema discovery & foreign key mapping
- [ ] Interactive Web UI
- [ ] Multi-turn conversational context & query refinement