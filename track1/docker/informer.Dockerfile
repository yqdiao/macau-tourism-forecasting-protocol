FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ARG UPSTREAM_COMMIT
RUN test -n "$UPSTREAM_COMMIT"
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/zhouhaoyi/Informer2020.git /opt/upstream \
 && git -C /opt/upstream checkout --detach "$UPSTREAM_COMMIT" \
 && test "$(git -C /opt/upstream rev-parse HEAD)" = "$UPSTREAM_COMMIT"
WORKDIR /workspace
COPY track1/docker/requirements-foundation-common.txt /tmp/common.txt
RUN pip install --no-cache-dir -r /tmp/common.txt
COPY track1/code/ ./track1/code/
COPY track1/configs/ ./track1/configs/
COPY shared/configs/ ./shared/configs/
ENV PYTHONPATH=/opt/upstream
ENTRYPOINT ["python", "track1/code/run_foundation_backtest.py"]
