# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

MAINTAINER hdinjos

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

ENV PORT 5000

WORKDIR $APP_HOME

COPY './requirements.txt' .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
 
RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

RUN pip3 install gunicorn

COPY . .

ENV PORT 5000
ENV HOST 0.0.0.0

EXPOSE 5000:5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]