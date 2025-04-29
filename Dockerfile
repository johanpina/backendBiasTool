FROM python:3.11-slim

# Evita preguntas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalación de dependencias del sistema

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias e instalar
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia del resto del código
COPY . .

# Puerto expuesto para FastAPI
EXPOSE 8000

# Comando por defecto
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]