# Инструкции по запуску микросервиса

### 1. FastAPI микросервис в виртуальном окружение

```
python3 -m venv ./.venv
source .venv/bin/activate
cd services/
python -m pip install -r requirements.txt
cd app
uvicorn real_estate_app:app --reload
```


### 2.1. FastAPI микросервис в Docker-контейнере
'''
docker pull python:3.11-slim 
docker image build . -f ./Dockerfile_ml_service --tag real_estate_price_predict:0
docker container run --publish 4601:8081 --volume=./models:/app/models   --env-file .env real_estate_price_predict:latest
'''

### 2.2. FastAPI микросервис с использованием Docker Compose
'''
cd services/app
docker pull python:3.11-slim
docker compose up  --build
'''