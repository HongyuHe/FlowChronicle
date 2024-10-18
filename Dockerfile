# syntax=docker/dockerfile:1
# https://docs.docker.com/develop/develop-images/multistage-build/


FROM continuumio/anaconda3:latest@sha256:8115a7841dc0f82e0a9c2d156239164c86f0b85c071d4dc8d957bd8f3d24e024

RUN apt update || true
RUN apt install -y gcc || true
RUN apt install -y gdb || true
RUN apt install -y nano htop wget tmux r-base || true 

RUN conda run -n base pip install scikit-mine || true

RUN conda install -c conda-forge ipympl || true
RUN conda install -c conda-forge eigen  || true
RUN conda install -c conda-forge pgmpy

RUN conda run -n base pip install saxpy || true
RUN conda run -n base pip install pybind11 || true
RUN conda run -n base pip install prettytable || true
RUN conda run -n base pip install wfdb || true
RUN conda run -n base pip install editdistance || true
RUN conda run -n base pip install bidict || true
RUN conda run -n base pip install pyinform || true
RUN conda run -n base pip install nptyping || true
RUN conda run -n base pip install opt-einsum
RUN conda run -n base pip install scikit-learn
