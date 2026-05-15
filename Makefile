PYTHON ?= python3
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
UI_HOST ?= localhost
UI_PORT ?= 8501
API_URL ?= http://$(API_HOST):$(API_PORT)

.PHONY: install run run-api run-ui

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run-api:
	$(PYTHON) -m uvicorn src.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

run-ui:
	API_URL=$(API_URL) UI_HOST=$(UI_HOST) UI_PORT=$(UI_PORT) $(PYTHON) -m src.ui.app

run:
	@$(PYTHON) -m uvicorn src.api.main:app --host $(API_HOST) --port $(API_PORT) --reload & \
	API_PID=$$!; \
	echo "API Server: http://$(API_HOST):$(API_PORT)"; \
	echo "Gradio UI: http://$(UI_HOST):$(UI_PORT)"; \
	trap 'kill $$API_PID' EXIT INT TERM; \
	API_URL=$(API_URL) UI_HOST=$(UI_HOST) UI_PORT=$(UI_PORT) $(PYTHON) -m src.ui.app
