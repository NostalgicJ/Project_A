# VERSION (PyTorch 2.3.0 / CUDA 12.1 / Ubuntu 22.04 기반으로 변경)
# 가장 최신 버전의 Ubuntu (22.04) 기반 이미지를 사용하여 네트워크 문제를 우회합니다.
ARG PYTORCH="2.3.0"
ARG CUDA="12.1"
ARG CUDNN="8"

# 1. 베이스 이미지 (Ubuntu 22.04 기반, Conda 이미 포함)
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV 변수 설정
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CONDA_DIR /opt/conda
ENV CMAKE_PREFIX_PATH $CONDA_DIR
ENV PATH $CONDA_DIR/bin:$PATH

# 2. apt-get 패키지 설치
# [trusted=yes] 등 EoL 관련 코드는 최신 Ubuntu 22.04이므로 모두 제거합니다.
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Conda 환경 자동 생성
WORKDIR /workspace
COPY text_recognition_package.yaml /workspace/text_recognition_package.yaml

# PyTorch 2.x 버전과 yaml 파일의 호환성을 위해 --force 옵션 추가
RUN conda env create --file text_recognition_package.yaml --force

# 4. 기본 셸 및 명령어 설정
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
