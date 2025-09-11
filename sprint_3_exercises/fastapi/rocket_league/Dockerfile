FROM python:3.10-slim

WORKDIR /app
#install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

#Copy the rest of the app
COPY . .

#Run the API using uvicorn (expose port)
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port", "8000"]