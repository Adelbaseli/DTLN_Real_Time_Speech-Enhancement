# Lightweight real-time-inference serving image: onnxruntime only, no torch.
# Build from the repo root:  docker build -f docker/Dockerfile -t dtln-serve .
FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

COPY src/onnx_infer.py src/onnx_infer.py
COPY src/__init__.py src/__init__.py
COPY serving/app.py serving/app.py
COPY onnx/ onnx/

ENV ONNX_DIR=/app/onnx
EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
