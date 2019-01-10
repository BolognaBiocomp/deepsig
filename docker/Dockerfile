# Base Image
FROM python:2.7.15-slim-stretch

# Metadata
LABEL base.image="python:2.7.15-slim-stretch"
LABEL version="0.9"
LABEL software="DeepSig"
LABEL software.version="2018012"
LABEL description="an open source software tool to predict signal peptides in proteins"
LABEL website="https://deepsig.biocomp.unibo.it"
LABEL documentation="http://deepsig.biocomp.unibo.it"
LABEL license="GNU GENERAL PUBLIC LICENSE Version 3"
LABEL tags="Proteomics"
LABEL maintainer="Castrense Savojardo <castrense.savojardo2@unibo.it>"

WORKDIR /usr/src/deepsig

COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    useradd -m deepsig

USER deepsig

COPY .. .

WORKDIR /data/

# Verbosity level of Tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=3 DEEPSIG_ROOT=/usr/src/deepsig PATH=/usr/src/deepsig:$PATH

ENTRYPOINT ["/usr/src/deepsig/deepsig.py"]