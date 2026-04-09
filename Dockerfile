# syntax=docker/dockerfile:1.4

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel AS tslib

WORKDIR /workspace

ARG http_proxy
ARG https_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV PYTHONPATH=/workspace:$PYTHONPATH

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

CMD ["bash"]