FROM python:3.12-bullseye

RUN apt update && apt upgrade -y

RUN mkdir /project


WORKDIR /project

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD []
