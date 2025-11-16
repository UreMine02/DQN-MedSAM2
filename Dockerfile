FROM nvcr.io/nvidia/pytorch:24.10-py3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 git ca-certificates && apt-get clean 
COPY requirements.txt /tmp/
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install -r /tmp/requirements.txt && pip install --upgrade google-cloud-storage 
RUN apt-get install -y tmux && apt-get install -y libevent-dev ncurses-dev build-essential bison pkg-config
