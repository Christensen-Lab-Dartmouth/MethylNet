FROM continuumio/miniconda3:4.5.11

RUN apt-get -y update

RUN apt-get -y install gcc

#RUN conda install -y -c conda-forge libgcc-ng=7.2.0 python=3.6.7

RUN pip install numpy==1.15.4 pandas==0.23.4

RUN conda install -y -c r r=3.5.1 r-cairo=1.5_9 r-devtools=1.13.6 python=3.6.7

#RUN conda install -y pip=10.0.1

RUN conda install -y scikit-learn=0.20.1 rpy2=2.9.1 sqlite=3.25.3 readline=7.0 click=6.7 cairo=1.14.12 python=3.6.7

RUN conda install -y -c conda-forge umap-learn=0.3.7 python=3.6.7

RUN conda install -y -c plotly plotly=3.4.2 python=3.6.7

RUN conda install -y -c pytorch pytorch=0.4.1 torchvision python=3.6.7

RUN apt-get -y install g++

RUN pip install fancyimpute==0.4.2

RUN conda install -y -c anaconda cudatoolkit=9.0

RUN pip install pandas==0.23.4

RUN conda install -y -c bioconda bioconductor-biocinstaller=1.30.0 bioconductor-geneplotter=1.58.0 bioconductor-geoquery=2.48.0

RUN mkdir /scripts/

COPY *.py /scripts/

RUN chmod 777 -R /scripts/

RUN python /scripts/preprocess.py install_custom -m -p sva

RUN python /scripts/preprocess.py install_r_packages -p knitr -p markdown -p gridExtra -p multcomp -p fastICA -p statmod -p lme4 -p BiocManager

RUN python /scripts/preprocess.py install_custom -p DNAcopy -p gdsfmt

RUN python /scripts/preprocess.py install_custom -p minfi -p ENmix

RUN python /scripts/preprocess.py install_meffil




RUN apt-get install -y xvfb

WORKDIR /root

#ENTRYPOINT ["/usr/bin/tini","-s","--"]
