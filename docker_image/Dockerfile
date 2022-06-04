FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV TZ=Asia/Tokyo 
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH
ENV CMAKE_PREFIX_PATH=/opt/conda/envs/tvm0.9
ENV CONDA_PREFIX=/opt/conda/envs/tvm0.9
ENV CXX=/usr/bin/clang++-10
ENV CC=/usr/bin/clang-10
ENV TVM_HOME=/root/tvm
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

ARG DOCKER_USER=docker

RUN apt-key del 3bf863cc && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get -y update
RUN apt-get -y upgrade

RUN apt-get install -y tree screen tmux tzdata openssh-server && \
    apt-get install -y vim wget bzip2 ca-certificates curl git unzip pkg-config && \
    apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev clang

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    /opt/conda/bin/conda clean --all && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

WORKDIR /root/.
RUN git config --global user.name ${DOCKER_USER} && \
    git config --global user.email ${DOCKER_USER}@example.com

RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.0/clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz && \
    tar -Jxvf clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz &&\
    git clone --recursive https://github.com/hirohaku21/TVMVisualize_AlexNet_Graph.git && \
    git clone --recursive https://github.com/pytorch/pytorch -b v1.7.0 && \
    git clone --recursive https://github.com/pytorch/vision.git -b v0.8.1 && \
    git clone --recursive https://github.com/apache/tvm.git && \
    git clone https://github.com/longcw/RoIAlign.pytorch.git && \
    sed -i 's/ -D_GLIBCXX_USE_CXX11_ABI=0//g' /root/tvm/cmake/modules/contrib/PT_TVMDSOOP.cmake && \
    sed -i 's/3.2/3.20/g' /root/tvm/CMakeLists.txt && \
    sed -i 's/3.2/3.20/g' /root/tvm/apps/pt_tvmdsoop/CMakeLists.txt && \
    cd tvm && \
    mkdir build && \
    cp cmake/config.cmake build/ && \
    sed -i -e 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g'  \
        -e 's/set(USE_GRAPH_EXECUTOR_CUDA_GRAPH OFF)/set(USE_GRAPH_EXECUTOR_CUDA_GRAPH ON)/g'  \
        -e 's/set(USE_LLVM OFF)/set(USE_LLVM "\/root\/clang+llvm-12.0.0-x86_64-linux-gnu-ubuntu-20.04\/bin\/llvm-config --link-static")/g' \
        -e 's/set(USE_CUDNN OFF)/set(USE_CUDNN ON)/g'  \
        -e 's/set(USE_CUBLAS OFF)/set(USE_CUBLAS ON)/g'  \
        -e 's/set(USE_PT_TVMDSOOP OFF)/set(USE_PT_TVMDSOOP ON)/g' build/config.cmake && \
    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> /root/tvm/build/config.cmake

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n tvm0.9 python=3.7.10 -y && \
    conda activate tvm0.9 && \
    conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y && \
    conda install -c pytorch magma-cuda111  && \
    pip install graphviz matplotlib decorator scipy attrs tornado xgboost==1.5.0 psutil cloudpickle pytest && \
    cd /root/pytorch && \
    python setup.py install && \
    cd /root/vision && \
    python setup.py install && \
    cd /root/tvm/build && \
    cmake .. && \
    make -j$(nproc)
