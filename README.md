# MethylNet

Deep Learning with Methylation

MethylNet is a command line tool and python library that provides classes to handle deep learning tasks for methylation data. It is built off of pythonic MethylationArray data types introduced in (https://github.com/Christensen-Lab-Dartmouth/PyMethylProcess), and uses PyTorch to explore/make predictions on the methylation data.  

https://www.biorxiv.org/content/10.1101/692665v1   

Help docs: https://christensen-lab-dartmouth.github.io/MethylNet/  


**What MethylNet can do:**  
1. Extract DNA Methylation Latent Space by training Variational Auto-encoders (VAE) after hyperparameter and neural network topology grid-search.  

![MethylNetPresentationSpring2019Lab 001](https://user-images.githubusercontent.com/19698023/55677380-32bb3d00-58b4-11e9-93bd-2cdc669bd6d8.jpeg)
![MethylNetPresentationSpring2019Lab 002](https://user-images.githubusercontent.com/19698023/55677381-32bb3d00-58b4-11e9-92ea-07d437a910e3.jpeg)
2. Make classification, single- and multi-output regression predictions on the methylation data such as age, cell type proportions, and disease state. This is done after transfer learning the VAE topology and a hyperparameter and neural network topology grid-search.  

![MethylNetPresentationSpring2019Lab 003](https://user-images.githubusercontent.com/19698023/55677389-436bb300-58b4-11e9-9bce-30d16bf71db1.jpeg)
3. Find most important CpGs for predictions using SHAP, interpreting predictions on the individual and aggragated class level.  

![MethylNetPresentationSpring2019Lab 004](https://user-images.githubusercontent.com/19698023/55677383-32bb3d00-58b4-11e9-9ecf-ab0eb135c740.jpeg)
4. Interrogate these extracted CpGs using popular pipelines such as LOLA, gometh, GSEA and overlap CpGs with other known sets of CpGs. In addition, overlap top sets of CpGs with other CpGs of other predictions.  

![MethylNetPresentationSpring2019Lab 005](https://user-images.githubusercontent.com/19698023/55677384-32bb3d00-58b4-11e9-9275-ee595fb81e0f.jpeg)

NOTE: Images will be updated to reflect latest set of MethylNet commands.   

MethylNet is currently in review, Wiki page in progress. Biorxiv can be accessed at: https://www.biorxiv.org/content/10.1101/692665v1   

Help docs: https://christensen-lab-dartmouth.github.io/MethylNet/  

**Install (Conda highly recommended, though Docker can be used):**
* conda create -n methylnet python=3.6   
* source activate methylnet   
* See install instructions for PyMethylProcess at https://github.com/Christensen-Lab-Dartmouth/PyMethylProcess  
* conda install pytorch torchvision -c pytorch  
* pip install methylnet  
* Alternative install: clone this repository and run python setup.py sdist bdist_wheel && pip install dist/methylnet-0.1.tar.gz   
* Run for GSEA collections: download_help_data   
* Alternative docker install: docker pull joshualevy44/methylnet:0.1     
                * If looking to use only CPUs, only core docker or singularity needed, and see singularity website for information on how to pull Docker images  
                * See https://singularity.lbl.gov/faq#does-singularity-support-containers-that-require-gpus for information on getting Singularity to work with GPUs  
                * nvidia-docker is also the Docker equivalent to run GPU jobs  
* GPU usage is also possible through the base install (without Docker, or considering nvidia-docker options), provided that your machine has GPU access.  

**Running MethylNet:**
1. source activate methylnet  
2. Pre-processing pipeline: Instructions available at https://github.com/Christensen-Lab-Dartmouth/PyMethylProcess to go from 450K/850K IDATs to MethylationArray datatypes that can easily be learned from.  
3. Run embedding hyperparameter scan  
4. Choose top embedding hyperparameters and train one last time again  
5. Run prediction hyperparameter scan  
6. Choose top prediction hyperparameters and train one last time again  
7. Find top CpGs by running SHAP.  
8. Interrogate SHAP derived CpGs.  
9. See help-docs for usage.  

**Running Test Pipeline:**
1. docker pull joshualevy44/methylnet:0.1  
2. Alternative: sh docker_build.sh  
3. docker run -it joshualevy44/methylnet:0.1  
4. Alternative: sh run_docker.sh  
5. methylnet-test test_pipeline  

**Example of Running Pipeline:**  
See ./example_scripts for examples on how to explicitly run:
![1](https://user-images.githubusercontent.com/19698023/55677358-f12a9200-58b3-11e9-8aaf-50536d2afb00.jpeg)
![2](https://user-images.githubusercontent.com/19698023/60547526-f306c480-9ced-11e9-817b-b4566edac22f.jpeg)
![3](https://user-images.githubusercontent.com/19698023/60547527-f306c480-9ced-11e9-9b0e-9ec89e056fa4.jpeg)
![4](https://user-images.githubusercontent.com/19698023/60547528-f306c480-9ced-11e9-9666-161a54a9e237.jpeg)
![5](https://user-images.githubusercontent.com/19698023/60547529-f306c480-9ced-11e9-8267-acb72940729b.jpeg)
![6](https://user-images.githubusercontent.com/19698023/60547530-f306c480-9ced-11e9-9295-31ae3b9c3edf.jpeg)
![7](https://user-images.githubusercontent.com/19698023/60547531-f306c480-9ced-11e9-9ddc-4c8c60853445.jpeg)
![8](https://user-images.githubusercontent.com/19698023/60547532-f306c480-9ced-11e9-8380-643b0e548d20.jpeg)
![9](https://user-images.githubusercontent.com/19698023/60547533-f306c480-9ced-11e9-84c6-2530018d7871.jpeg)
