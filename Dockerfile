FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 4000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:4000", "app:app"]

# CMD ["python", "app.py"]
