FROM python:3.10-slim

WORKDIR /app

COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "source.py"]
