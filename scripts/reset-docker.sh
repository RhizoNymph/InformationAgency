#!/bin/bash

# This script stops and removes all containers, networks, and volumes
# defined in the docker-compose.yaml file in the parent directory.

echo "Stopping and removing containers, networks, and volumes..."
docker compose down --volumes --remove-orphans

echo "Docker environment reset complete."
