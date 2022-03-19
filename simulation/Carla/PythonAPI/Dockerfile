FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update
RUN apt install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y --no-install-recommends \
	build-essential \
	git \
	curl \
	libsdl2-dev \ 
	libsdl2-image-dev \
	libsdl2-mixer-dev \
	libsdl2-ttf-dev \
	libfreetype6-dev \
	libportmidi-dev \
	libjpeg-dev \
	python3-setuptools \
	python3.7 \
	python3-dev \
	python3-numpy \
	python3-pip \
	python3-tk 

WORKDIR /tmp
COPY requirements_deep_planning.txt /tmp/requirements_deep_planning.txt
COPY requirements_carla.txt /tmp/requirements_carla.txt
COPY ./carla/dist/carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl /tmp/carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install setuptools
RUN python3.7 -m pip install -r requirements_deep_planning.txt
RUN python3.7 -m pip install -r requirements_carla.txt
RUN python3.7 -m pip install carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl


ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
