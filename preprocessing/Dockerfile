FROM python:3.9-slim-bullseye AS build

RUN  apt-get update \
  && apt-get install -y wget


ENV PYTHONUNBUFFERED 1

COPY preprocessing /preprocessing 

ENV CONDA_DIR /root/conda
RUN apt-get install wget
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
RUN bash Anaconda3-2022.05-Linux-x86_64.sh -b -p $CONDA_DIR

ENV PATH=$CONDA_DIR/bin:$PATH

RUN $CONDA_DIR/bin/conda shell.bash hook && conda init bash && conda create -n deeptyper_3.6 python=3.6

