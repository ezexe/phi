# The image the release workflow pushes to ghcr.io/ezexe/phi: phi-prune from the wheel of the release, on torch's
# CPU-only wheels. The build context is the folder that holds the wheel, as in `docker build -f lbiza/Dockerfile dist`.
FROM python:3.14-slim

LABEL org.opencontainers.image.source="https://github.com/ezexe/phi"
LABEL org.opencontainers.image.description="φ-pruning under the (1,∞)-RLL constraint, Fibonacci coding of quantized weights, and the ranking models the p4 plugins train in a project's store"
LABEL org.opencontainers.image.licenses="Apache-2.0"

COPY *.whl /tmp/wheel/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /tmp/wheel/*.whl \
    && rm -r /tmp/wheel

ENTRYPOINT ["phi-prune"]
