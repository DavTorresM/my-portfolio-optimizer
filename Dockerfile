# Usa una imagen oficial de Python
FROM python:3.10

# Establece el directorio de trabajo
WORKDIR /app

# Copia e instala dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copia el código de la aplicación
COPY main.py .
COPY helpers.py .

# Expone el puerto requerido por Cloud Run
EXPOSE 8080

# Comando de inicio para apps FastAPI con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]