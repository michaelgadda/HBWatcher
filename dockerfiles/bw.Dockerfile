FROM ubuntu:26.04

WORKDIR /bw/

RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
DEBIAN_FRONTEND=noninteractive apt-get -y  install ffmpeg libgl1 libpq-dev build-essential python3.14 python3.14-dev python3.14-venv python3-pip

COPY ./requirements.txt /bw/requirements.txt

RUN python3 -m venv /opt/myvenv

ENV PATH="/opt/myvenv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/myvenv/"

RUN pip install -r /bw/requirements.txt

