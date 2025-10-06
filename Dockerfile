
# Usar una imagen base más robusta que incluye herramientas de compilación
FROM python:3.9-bullseye

# Instalar las dependencias de sistema necesarias para compilar paquetes como lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos e instalar las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto que usará la aplicación
EXPOSE 8000

# Comando para iniciar el servidor de Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
