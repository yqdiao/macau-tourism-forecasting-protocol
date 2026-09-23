FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ARG UPSTREAM_COMMIT
RUN test -n "$UPSTREAM_COMMIT"
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/google-research/timesfm.git /opt/upstream \
 && git -C /opt/upstream checkout --detach "$UPSTREAM_COMMIT" \
 && test "$(git -C /opt/upstream rev-parse HEAD)" = "$UPSTREAM_COMMIT" \
 && pip install --no-cache-dir "/opt/upstream[torch]"
WORKDIR /app
COPY docker/requirements-foundation-common.txt /tmp/common.txt
RUN pip install --no-cache-dir -r /tmp/common.txt
COPY data_pipeline.py run_foundation_backtest.py ./
COPY foundation_models/ ./foundation_models/
COPY configs/ ./configs/
ENTRYPOINT ["python", "run_foundation_backtest.py"]
