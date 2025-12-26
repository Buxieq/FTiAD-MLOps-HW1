.PHONY: build-push test lint help

# Docker image configuration
DOCKER_USERNAME ?= buxieq
IMAGE_NAME ?= mlops-hw1
IMAGE_TAG ?= latest
FULL_IMAGE_NAME = $(DOCKER_USERNAME)/$(IMAGE_NAME):$(IMAGE_TAG)

help:
	@echo "Available commands:"
	@echo "  make build-push  - Build and push Docker image"
	@echo "  make test        - Run unit tests"
	@echo "  make lint        - Run linters"

build-push:
	@echo "Building $(FULL_IMAGE_NAME)"
	docker build -t $(FULL_IMAGE_NAME) .
	@echo "Pushing"
	docker push $(FULL_IMAGE_NAME)
	@echo "Pushed"

test:
	@echo "unit tests"
	pytest tests/ -v --tb=short

lint:
	@echo "Running linters"
	@echo "ruff"
	ruff check . || true
	@echo "Formatting check with ruff"
	ruff format --check . || true
	@echo "Linting is complete"

