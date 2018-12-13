FROM methylnet:0.1

RUN conda install -y curl=7.62.0 openssl=1.1.1 python=3.6.7

RUN conda install -y -c conda-forge matplotlib=3.0.2 pathos=0.2.1 python=3.6.7

RUN python /scripts/installer.py install_r_packages -p openssl

RUN python /scripts/installer.py install_custom -m -p rtracklayer=3.8

copy *.py /scripts/

ENTRYPOINT ["/usr/bin/tini","-s","--"]
