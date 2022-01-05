FROM tensorflow/tensorflow:latest-gpu

RUN python3 -m pip install --upgrade pip

WORKDIR /cellx
COPY . /cellx
RUN pip install -r requirements.txt
RUN pip install -e .
