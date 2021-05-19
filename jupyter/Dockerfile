FROM python:3.7-slim
WORKDIR /code
ENV DEBIAN_FRONTEND=noninteractive 
RUN pip3 install -U pip
RUN pip3 install jupyterlab tqdm scikit-learn requests numpy matplotlib scipy
EXPOSE 8888
COPY . .
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# RUN make html
# CMD ["jupyter", "lab"]
