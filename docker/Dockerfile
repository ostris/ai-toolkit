FROM runpod/base:0.6.2-cuda12.1.0
LABEL authors="jaret"

# Install dependencies
RUN apt-get update

WORKDIR /app
ARG CACHEBUST=1
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git submodule update --init --recursive

WORKDIR /app/ai-toolkit

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install -r requirements.txt

RUN apt-get install -y tmux nvtop htop

WORKDIR /
CMD ["/start.sh"]