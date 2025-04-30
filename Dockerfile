FROM python:3.9-slim

# Evita preguntas interactivas y asegúrate de tener libgomp
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalación de dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia del código del backend
COPY . .

# Puerto expuesto para FastAPI
EXPOSE 8000

# Arranca la app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]