FROM joshualevy44/pymethylprocess:0.1.3

RUN apt-get install -y python-setuptools

RUN pip3 install torch torchvision

RUN mkdir ./methylnet_code

COPY . ./methylnet_code/

RUN pip3 install --upgrade setuptools

RUN pip3 install pandas==0.24.1

RUN cd ./methylnet_code && python3 setup.py sdist bdist_wheel && pip install dist/methylnet-0.1.tar.gz

RUN tar -xzf ./methylnet_code/test_data/age_test_data.tar.gz -C /pymethyl/

RUN cp ./methylnet_code/example_scripts/*.yaml /pymethyl/

RUN chmod 755 -R train_val_test_sets

WORKDIR /pymethyl
