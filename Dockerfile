# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTE=1
ENV UV_LINK_MODE=copy

# Copy uv binary (for dependency management)
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Add virtual environment path to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Layer 1: Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Copy dependency files
COPY ./pyproject.toml ./uv.lock ./.python-version /app/

# Layer 3: Install Python dependencies
RUN uv sync --frozen --no-install-project

# Layer 4: Copy YOLO model files (for caching)
COPY ./yolov8n-pose.pt ./yolov8n.pt /app/

# Layer 5: Copy app source code
COPY ./app /app/app

# Expose the default FastAPI port
EXPOSE 8000

# Use uvicorn to run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
