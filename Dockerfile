FROM methylnet:0.1

copy installer.py /scripts/

RUN apt-get install -y libopenmpi-dev

RUN conda install -y curl=7.62.0 openssl=1.1.1 python=3.6.7

RUN conda install -y -c conda-forge matplotlib=3.0.2 mpi=1.0 openmpi=3.1.2 pathos=0.2.1 python=3.6.7

RUN pip install pyina==0.2.0

RUN python /scripts/installer.py install_r_packages -p openssl

RUN python /scripts/installer.py install_custom -m -p rtracklayer=3.8 -p GEOquery=3.8

RUN conda install -y -c menpo ffmpeg=3.1.3

copy *.py /scripts/

ENTRYPOINT ["/usr/bin/tini","-s","--"]
