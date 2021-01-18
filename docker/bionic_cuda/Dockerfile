FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    && apt-get install --no-install-recommends -y wget nano cmake build-essential git \
    ca-certificates libomp-dev libgtest-dev libpcl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /root/fast_gicp
WORKDIR /root/fast_gicp
RUN git submodule init && git submodule update

RUN rm -rf /root/fast_gicp/build
WORKDIR /root/fast_gicp/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_VGICP_CUDA=ON -DBUILD_apps=OFF -DBUILD_test=OFF
RUN make -j$(nproc)

WORKDIR /root

CMD ["bash"]
