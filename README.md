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
### Configure Environment

- Copy the template and adjust values:
  ```bash
  cp env.sample .env
  ```
- Fill in API keys for OpenAI, Langfuse, Qdrant, and **Google GenAI** (`GOOGLE_GENAI_API_KEY`).
- Align base URLs with your deployment:
  - Local processes: `BASE_HTTP_URL=http://localhost:8000`, `WS_URL=ws://localhost:8000`
  - Docker Compose: `BASE_HTTP_URL=http://app:8000`, `WS_URL=ws://app:8000`
- Increase `INGEST_HTTP_TIMEOUT` if ingestion of large sources exceeds 60 s.
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

## Video Generation Flow

`POST /video/generate` accepts a multipart request that combines metadata and one or more product images:

1. Add a form field named `payload` containing a JSON string compatible with `VideoGenerationRequest`.
2. Attach one or more files under the `product_images` field (PNG or JPEG).

Example curl:
```bash
curl -X POST http://localhost:8000/video/generate \
  -F 'payload={
        "business_type": "luxury travel agency",
        "product_description": "Curated weekend escapes to Abu Dhabi with VIP experiences.",
        "aspect_ratio": "16:9",
        "duration_seconds": 12,
        "creative_direction": "Energetic yet refined, showcasing skyline and premium amenities.",
        "negative_prompt": "Avoid gloomy tones",
        "extra_instructions": "Include smooth transitions and a final call-to-action screen."
      };type=application/json' \
  -F 'product_images=@/path/to/abu-dhabi-sunset.jpg' \
  -F 'product_images=@/path/to/luxury-suite.jpg'
```

The response includes a `video_uri` returned by Google along with the raw payload for debugging. Set `GOOGLE_GENAI_API_KEY` before invoking the endpoint.

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




