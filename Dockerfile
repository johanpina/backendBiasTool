# Imagen ligera: al eliminar Aequitas (y su cadena lightgbm/fairgbm) ya no se
# necesita build-essential ni compilación de paquetes nativos.
FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias de Python (solo runtime, sin herramientas de compilación).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación.
COPY . .

EXPOSE 8000

# Servidor de producción (sin --reload, que es solo para desarrollo).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
