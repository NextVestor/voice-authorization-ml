# FROM python:3.7-slim
# WORKDIR /code
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0
# RUN apk add --no-cache gcc musl-dev linux-headers openblas-dev musl-dev g++ python3.7-dev
# COPY requirements.txt requirements.txt
# # RUN pip install --upgrade pkg-config
# RUN pip install -r requirements.txt
# EXPOSE 5000
# COPY . .
# CMD ["flask", "run"]

FROM ubuntu:18.04
WORKDIR /code
RUN apt-get update
RUN apt-get update && apt-get install -y python3-pip python3.7-dev python-pip \
    libxft-dev libfreetype6 libfreetype6-dev gcc musl-dev libopenblas-dev g++
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run"]
