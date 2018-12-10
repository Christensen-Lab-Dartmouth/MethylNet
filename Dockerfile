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

RUN conda install -y -c bioconda bioconductor-biocinstaller=1.30.0 bioconductor-geneplotter=1.58.0

RUN mkdir /scripts/

COPY *.py /scripts/

RUN chmod 755 -R /scripts/

RUN python /scripts/preprocess.py install_r_packages -p BiocManager -p remotes -p knitr -p markdown -p gridExtra -p multcomp -p fastICA -p statmod -p lme4 -p Cairo -p BiocManager

RUN python /scripts/preprocess.py install_custom -m -p GEOquery=3.8 -p sva -p S4Vectors -p DNAcopy -p gdsfmt -p ENmix

RUN conda install -y -c conda-forge unzip=6.0 xorg-libx11=1.6.6 python=3.6.7

RUN apt-get install -y libcairo2-dev libxt-dev

RUN python /scripts/preprocess.py install_custom -m -p illuminaio

RUN  wget https://github.com/perishky/meffil/archive/master.zip && unzip master.zip && mv meffil-master meffil && R CMD INSTALL meffil

#python /scripts/preprocess.py install_meffil

RUN apt-get install -y xvfb

RUN conda install -y -c conda-forge tar=1.29 python=3.6.7

RUN conda install -y -c ostrokach gzip=1.7 python=3.6.7

WORKDIR /root

ENTRYPOINT ["/usr/bin/tini","-s","--"]
