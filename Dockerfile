FROM methylnet:0.1

RUN conda install -y curl=7.62.0 openssl=1.1.1

copy *.py /scripts/

ENTRYPOINT ["/usr/bin/tini","-s","--"]
