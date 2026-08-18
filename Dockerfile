# Imagen ligera: sin Aequitas ya no hace falta build-essential ni compilación nativa.
FROM python:3.12-slim

# Buenas prácticas de runtime Python en contenedor.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Capa de dependencias cacheable (solo se reconstruye si cambia requirements.txt).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código de la aplicación (los tests/docs/notebook se excluyen vía .dockerignore).
COPY . .

# Ejecutar como usuario no-root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Verificación de salud contra el endpoint /api/health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
