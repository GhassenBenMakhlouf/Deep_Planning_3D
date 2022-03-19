FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	git \
	curl \	
	libglib2.0-0 \
	software-properties-common \
	python3.9 \
	python3.9-dev \
	python3-pip \
	python3-tk 

WORKDIR /tmp
COPY requirements.txt /tmp/requirements.txt

RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install setuptools
RUN python3.9 -m pip install -r requirements.txt
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2


ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
