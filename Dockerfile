# 1. Base image
FROM python:3.11-slim

# 2. Workdir inside container
WORKDIR /app

# 3. Install system deps (optional but useful for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency file first (for better caching)
COPY requirements.txt .

# 5. Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the project
COPY . .

# 7. Expose FastAPI port
EXPOSE 8000

# 8. Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]