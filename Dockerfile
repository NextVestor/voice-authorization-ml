FROM ubuntu:18.04
WORKDIR /code
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update
RUN apt-get update && apt-get install -y python3-pip python3.7-dev python-pip \
    libxft-dev libfreetype6 libfreetype6-dev gcc musl-dev libopenblas-dev g++ \
    llvm-9 libsndfile1 python3.7-tk
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip3 install -U pip
RUN pip3 install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt
EXPOSE 5000
COPY . .
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# RUN make html
# CMD ["flask", "run"]
