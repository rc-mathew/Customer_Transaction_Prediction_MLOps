FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt



COPY src ./src
COPY models/customer_transaction_model.joblib ./models/customer_transaction_model.joblib

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]