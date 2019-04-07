python basic_installer.py -p Rcpp simpleCache BiocManager devtools Cairo remotes tidyverse knitr markdown gridExtra multcomp fastICA statmod lme4 base64enc ggpubr forcats
python basic_installer.py -b -p minfi ENmix geneplotter IlluminaHumanMethylation450kanno.ilmn12.hg19
python basic_installer.py -b -p sva S4Vectors DNAcopy gdsfmt illuminaio ggbio
python basic_installer.py -b -p rtracklayer GEOquery LOLA limma missMethyl TCGAbiolinks
python basic_installer.py -g perishky/meffil

module load python/3-Anaconda
conda create -y -n methylnet_pro2 python=3.6.7
source activate methylnet_pro2
python -m pip install numpy==1.15.4
conda install -y -c conda-forge unzip=6.0 xorg-libx11=1.6.6 tar=1.29 python=3.6.7
conda install -y curl=7.62.0 python=3.6.7
conda install -y sqlite=3.25.3 readline=7.0 cairo=1.14.12 python=3.6.7 # scikit-learn=0.20.1
conda install -y -c click click=6.7
python -m pip install rpy2
python -m pip install kneed
python -m pip install Cython pathos
python -m pip install nevergrad
#python -m pip download rpy2==2.9.5 && tar xvzf rpy2-2.9.5.tar.gz
#cd rpy2-2.9.5 && python setup.py build --r-home /dartfs-hpc/rc/home/w/f003k8w/R/x86_64-redhat-linux-gnu-library/3.5 install

conda install -y -c conda-forge umap-learn=0.3.7 python=3.6.7
conda install -y -c plotly plotly=3.4.2 python=3.6.7
conda install -y -c pytorch pytorch=0.4.1 torchvision python=3.6.7
conda install -y -c anaconda cudatoolkit=9.0
python -m pip install fancyimpute==0.4.2 pandas==0.23.4
python -m pip install shap matplotlib seaborn mlxtend
python -m pip install git+https://github.com/scikit-learn/scikit-learn.git@iterativeimputer
conda install -y -c menpo ffmpeg=3.1.3
