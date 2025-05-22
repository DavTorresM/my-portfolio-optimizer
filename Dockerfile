# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY main.py .
COPY helpers.py .

# Expose port required by Cloud Run
EXPOSE 8080

# Startup command for FastAPI apps with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]