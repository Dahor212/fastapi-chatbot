# Používáme oficiální Python obraz
FROM python:3.9-slim

# Nastavení pracovního adresáře v Dockeru
WORKDIR /app

# Kopírování requirements.txt do kontejneru
COPY requirements.txt /app/

# Instalace závislostí
RUN pip install --no-cache-dir -r requirements.txt

# Kopírování zbytku aplikace do kontejneru
COPY . /app/

# Nastavení příkazu pro spuštění aplikace
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
