# Base Image
FROM continuumio/miniconda3

# Metadata
LABEL base.image="continuumio/miniconda3"
LABEL version="0.9"
LABEL software="DeepSig"
LABEL software.version="2018012"
LABEL description="an open source software tool to predict signal peptides in proteins"
LABEL website="https://deepsig.biocomp.unibo.it"
LABEL documentation="https://deepsig.biocomp.unibo.it"
LABEL license="GNU GENERAL PUBLIC LICENSE Version 3"
LABEL tags="Proteomics"
LABEL maintainer="Castrense Savojardo <castrense.savojardo2@unibo.it>"

WORKDIR /usr/src/deepsig

COPY . .

WORKDIR /data/

RUN conda install --yes keras biopython

# Verbosity level of Tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=3 DEEPSIG_ROOT=/usr/src/deepsig PATH=/usr/src/deepsig:$PATH

ENTRYPOINT ["/usr/src/deepsig/deepsig.py"]
