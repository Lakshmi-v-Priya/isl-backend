# ---------- BASE IMAGE ----------
FROM python:3.10-slim

# ---------- SYSTEM DEPENDENCIES ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- WORKING DIRECTORY ----------
WORKDIR /app

# ---------- COPY DEPENDENCIES ----------
COPY requirements.txt .

# ---------- INSTALL PYTHON DEPENDENCIES ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- COPY APP CODE ----------
COPY app ./app

# ---------- EXPOSE PORT ----------
EXPOSE 8000

# ---------- START SERVER ----------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
