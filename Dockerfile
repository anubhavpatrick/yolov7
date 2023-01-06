# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /flask-yolo

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# For setting time zone
#https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -y && apt install ffmpeg libsm6 libxext6  -y

RUN pip3 install Flask
RUN pip3 install Flask-Bootstrap

COPY . .

CMD [ "python3", "-m" , "flask", "--app", "flaskApp", "run", "--host=0.0.0.0"]