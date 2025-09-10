FROM python:3.10.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Expose app port
EXPOSE 8000

# Define volume at /app/data to store the predictions database
VOLUME /app/data

# Run app on port 8000
CMD ["fastapi", "run", "main.py"]
