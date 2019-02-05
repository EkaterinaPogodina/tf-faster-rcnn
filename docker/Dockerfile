# mbuckler/tf-faster-rcnn-deps
#
# Dockerfile to hold dependencies for the Tensorflow implmentation
# of Faster RCNN

#FROM nvidia/cuda:9.0-cudnn7-devel
#WORKDIR /root

#RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
#		bc \
#		time \
#		build-essential \
#		cmake \
#		curl \
#		g++ \
#		gfortran \
#		git \
#		libffi-dev \
#		libfreetype6-dev \
#		libhdf5-dev \
#		libjpeg-dev \
#		liblcms2-dev \
#		libopenblas-dev \
#		liblapack-dev \
#		libopenjpeg5 \
#		libpng12-dev \
#		libssl-dev \
#		libtiff5-dev \
#		libwebp-dev \
#		libzmq3-dev \
#		nano \
#		pkg-config \
#		python-dev \
#		software-properties-common \
#		unzip \
#		vim \
#		wget \
#		zlib1g-dev \
#		qt5-default \
#		libvtk6-dev \
#		zlib1g-dev \
#		libjpeg-dev \
#		libwebp-dev \
#		libpng-dev \
#		libtiff5-dev \
#		libjasper-dev \
#		libopenexr-dev \
#		libgdal-dev \
#		libdc1394-22-dev \
#		libavcodec-dev \
#		libavformat-dev \
#		libswscale-dev \
#		libtheora-dev \
#		libvorbis-dev \
#		libxvidcore-dev \
#		libx264-dev \
#		yasm \
#		libopencore-amrnb-dev \
#		libopencore-amrwb-dev \
#		libv4l-dev \
#		libxine2-dev \
#		libtbb-dev \
#		libeigen3-dev \
#		python-dev \
#		python-tk \
#		python-numpy \
#		python3-dev \
#		python3-tk \
#		python3-numpy \
#		python3-pip \
#		ant \
#		default-jdk \
#		doxygen \
#		&& \
#	apt-get clean && \
#	apt-get autoremove && \
#	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
#	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3                  
# Get required python modules

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

ADD tf-faster-rcnn /root/

# Add CUDA to the path
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV CUDA_HOME /usr/local/cuda
RUN ldconfig

