module load python/3-Anaconda
conda create -y -n methylnet python=3.6.7
source activate methylnet
#conda install -y -c r mro-base
python -m pip install numpy==1.15.4
conda install -y gcc_linux-64 gxx_linux-64 gfortran_linux-64
conda install -y -c conda-forge libstdcxx-ng
conda install -y -c conda-forge unzip=6.0 xorg-libx11=1.6.6 tar=1.29 python=3.6.7
conda install -y curl=7.62.0 openssl=1.1.1 python=3.6.7

#conda install -y -c conda-forge r=3.5.1 # gcc r-rcpp
conda install -y -c r r=3.5.1 r-cairo=1.5_9 r-devtools=1.13.6 python=3.6.7
#conda install -c r _r-mutex=1.0.0-anacondar
conda install -y scikit-learn=0.20.1 rpy2=2.9.1 sqlite=3.25.3 readline=7.0 click=6.7 cairo=1.14.12 python=3.6.7
conda install -y -c conda-forge umap-learn=0.3.7 python=3.6.7
conda install -y -c plotly plotly=3.4.2 python=3.6.7
conda install -y -c pytorch pytorch=0.4.1 torchvision python=3.6.7
conda install -y -c anaconda cudatoolkit=9.0
#conda install -y -c bioconda bioconductor-biocinstaller=1.30.0 bioconductor-geneplotter=1.58.0
python -m pip install fancyimpute==0.4.2 pandas==0.23.4
#conda install -y -c conda-forge matplotlib=3.0.2 mpi=1.0 openmpi=3.1.2 pathos=0.2.1 python=3.6.7
conda install -y -c r -f r-base
python installer.py install_r_packages -p BiocManager -p Rcpp -p simpleCache
python installer.py install_custom -m -p minfi -p ENmix -p geneplotter -p IlluminaHumanMethylation450kanno.ilmn12.hg19
wget https://github.com/perishky/meffil/archive/master.zip && unzip master.zip && mv meffil-master meffil && R CMD INSTALL meffil
#python installer.py install_r_packages -p openssl
python installer.py install_custom -m -p rtracklayer=3.8 -p GEOquery=3.8
python installer.py install_custom -m -p LOLA -p limma -p missMethyl
python -m pip install shap matplotlib
conda install -y -c menpo ffmpeg=3.1.3
# conda clean --packages
