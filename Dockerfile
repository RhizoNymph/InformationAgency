# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set uv specific environment variable for non-interactive installs
ENV UV_NO_INTERACTION=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages (e.g., for OCR or PDF processing)
# Add any other required build-time or run-time OS packages here
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install uv using pip, which comes with the base Python image
# We install it separately so this layer can be cached effectively
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# Copy only project configuration files first to leverage Docker cache
# Include pyproject.toml and potentially uv.lock if you use it
COPY pyproject.toml ./
# COPY uv.lock ./ # Uncomment this line if you are using a uv.lock file

# Install Python dependencies using uv
# If you use `uv lock` and commit `uv.lock`: use `uv sync` for reproducible builds
# RUN uv sync --no-cache --strict # Uncomment this line to use uv.lock

# If you don't use a lock file, install directly from pyproject.toml:
# Use --no-cache to reduce image size
RUN uv pip install --no-cache --system -r pyproject.toml

# Copy the rest of the application code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run your application
# Use 0.0.0.0 to allow connections from outside the container within the Docker network
CMD ["uvicorn", "orchestrator.api:app", "--host", "0.0.0.0", "--port", "8000"]