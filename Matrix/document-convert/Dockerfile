FROM registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.2-cudnn8.2-trt8.0
WORKDIR /workspace
COPY . .
RUN apt-get update && apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg 
RUN /usr/bin/python -m pip install -r requirements.txt
RUN /usr/bin/python -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN /usr/bin/python -m pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html