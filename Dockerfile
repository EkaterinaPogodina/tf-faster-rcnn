FROM nvidia/cuda:9.0-cudnn7-devel
WORKDIR /root

# Get required packages
RUN apt-get update && \
  apt-get install vim \
                  python3-pip \
                  python3-dev \
                  python-opencv \
                  python3-tk \
                  libjpeg-dev \
                  libfreetype6 \
                  libfreetype6-dev \
                  zlib1g-dev \
                  cmake \
                  wget \
                  cython \
                  git \
		  time \
                  -y

RUN pip3 install --upgrade pip

# Update numpy
RUN pip3 install -U numpy

RUN pip install virtualenv
RUN virtualenv -p python3 virtual

WORKDIR /root

ADD . /root/

# Add CUDA to the path
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV CUDA_HOME /usr/local/cuda
RUN ldconfig

