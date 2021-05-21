FROM tensorflow/tensorflow:latest-gpu

WORKDIR /cellx
COPY . /cellx
RUN pip install -e .
