#!/bin/bash

echo "Starting LIA Resumo Semanal Service on Railway..."

export PORT=${PORT:-8000}

echo "Starting FastAPI server on port $PORT..."

uvicorn main:app --host 0.0.0.0 --port $PORT
