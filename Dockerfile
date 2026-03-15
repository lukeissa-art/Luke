FROM python:3.11-slim

# Install system deps for ssl/certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to the trading loop; override CMD to run the API instead.
CMD ["python", "run_bot.py"]
