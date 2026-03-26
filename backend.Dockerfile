FROM python:3.10-slim

WORKDIR /app

COPY . /app

EXPOSE 8080

CMD ["python", "-u", "server.py"]
