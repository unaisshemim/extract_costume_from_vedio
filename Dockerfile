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

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY ./pyproject.toml ./uv.lock ./.python-version /app/

# Install dependencies
RUN uv sync

# Copy app source code
COPY ./app /app/app

# Expose the default FastAPI port
EXPOSE 8000

# Use uvicorn to run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
