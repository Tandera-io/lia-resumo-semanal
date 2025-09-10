#!/bin/bash

echo "Starting LIA Resumo Semanal Service on Railway..."

export PORT=${PORT:-8001}

echo "Starting FastAPI server on port $PORT..."

python main.py
