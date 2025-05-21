FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python", "scripts/cli.py"]
# CMD ["python", "scripts/cli.py", "serve"] 