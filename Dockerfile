FROM continuumio/miniconda3:latest

RUN apt-get -y update

RUN apt-get -y install build-essential

RUN pip install numpy pandas

RUN conda install -y -c r r=3.5.1 r-cairo=1.5_9 r-devtools=1.13.6

RUN conda install -y pip=9.0.3

RUN conda install -y scikit-learn=0.20.1 rpy2=2.9.1 sqlite=3.25.3 readline=7.0 click=6.7 cairo=1.14.12

RUN conda install -y -c conda-forge umap-learn=0.3.7

RUN conda install -y -c plotly plotly=3.4.2

RUN conda install -y -c pytorch pytorch=0.4.1 torchvision

RUN pip install fancyimpute GEOparse

RUN mkdir /scripts/

COPY *.py /scripts/

RUN python /scripts/preprocess.py install_all_deps

RUN apt-get install -y xvfb

ENTRYPOINT ["/usr/bin/tini","-s","--"]
