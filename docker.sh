#!/bin/bash

# Define the supported commands
COMMANDS="up down build ingest logs"

# Check if a command is provided
if [ -z "$1" ]; then
    echo "Usage: ./docker.sh {up|down|build|ingest|logs}"
    exit 1
fi

case "$1" in
  up)
    echo "Starting services..."
    docker compose up -d
    echo "Services started. Frontend at http://localhost:3000, Backend at http://localhost:8001"
    ;;
  down)
    echo "Stopping services..."
    docker compose down
    ;;
  build)
    echo "Building images..."
    docker compose build
    ;;
  ingest)
    echo "Running ingestion..."
    # Run the ingestion script inside the backend container context
    # Use --rm to remove the container after execution
    docker compose run --rm backend python ingest.py
    ;;
  logs)
    echo "Showing logs..."
    docker compose logs -f
    ;;
  *)
    echo "Invalid command: $1"
    echo "Usage: ./docker.sh {up|down|build|ingest|logs}"
    exit 1
    ;;
esac
