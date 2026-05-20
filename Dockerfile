FROM python:3.11-slim

# Evitar que Python escriba archivos .pyc ocultos
ENV PYTHONDONTWRITEBYTECODE=1

# Forzar a que los logs se muestren en tiempo real en la consola de Railway
ENV PYTHONUNBUFFERED=1

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del proyecto
COPY . .

# Comando para arrancar el agente
# (Asegúrate de que tu archivo de Python se llame Agente.py, o cámbialo aquí)
CMD ["python", "Agente.py"]