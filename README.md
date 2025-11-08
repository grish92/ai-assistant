# AI Assistant

AI Assistant is a LangChain-based application that provides news-oriented conversational responses and structured summaries. It integrates with Langfuse for observability, Qdrant for vector search, and OpenAI for language generation. The project ships both a FastAPI backend and an optional Streamlit UI.



## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized setup)
- Access to:
  - OpenAI API key
  - Langfuse public/secret keys
  - Qdrant instance (local via Docker, or a managed instance)

### Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



## Running the Services

### Docker Compose

The easiest way to launch the stack locally:

```bash
docker compose up --build
```

```bash
docker compose down
```

### Local Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

For the Streamlit client:

```bash
streamlit run app/streamlit/streamlit_app.py
```

---

## Langfuse Integration

- Tracing is controlled via `LANGFUSE_TRACING_ENABLED`.
- `LangfuseCallbackHandler` logs each LLM invocation; spans show prompt inputs, outputs, and errors.
- Prompts can be stored in Langfuse; `prompt_manager` will fetch them by name and fall back to local YAML.
- Use `test_langfuse_integration.py` (see [Testing](#testing-and-debugging)) to verify connectivity.

---

## Prompts and Prompt Management

- Local prompt definitions live in `app/prompts/prompts.yaml`.
- Each entry may specify:
  - `description`: documentation
  - `langfuse_prompt`: Langfuse prompt identifier to fetch dynamically
  - `template`: fallback string used when Langfuse fetch fails
- The `prompt_manager` abstracts loading so application code simply calls `get_prompt_template("chat_response")`, etc.

---


echo "# ai-assistant" >> README.md
git init
git add -A
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/grish92/ai-assistant.git
git push -u origin main



# ai-assistant
