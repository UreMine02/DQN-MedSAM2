FROM nvcr.io/nvidia/pytorch:23.08-py3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 git ca-certificates && apt-get clean 
COPY requirements.txt /tmp/
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r /tmp/requirements.txt && pip install --upgrade google-cloud-storage 
RUN apt-get install -y tmux && apt-get install -y libevent-dev ncurses-dev build-essential bison pkg-config
