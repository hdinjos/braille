# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

MAINTAINER hdinjos

WORKDIR /app

COPY './requirements.txt' .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
 
RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "app.py"]
