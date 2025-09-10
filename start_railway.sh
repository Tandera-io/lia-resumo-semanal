#!/bin/bash


echo "Starting LIA Resumo Semanal Service on Railway..."

export PORT=${PORT:-8001}

echo "Starting FastAPI server on port $PORT..."
python -m uvicorn main:app --host 0.0.0.0 --port $PORT
