# MethylNet

Deep Learning with Methylation

MethylNet is a command line tool and python library that provides classes to handle deep learning tasks for methylation data. It is built off of pythonic MethylationArray data types introduced in (https://github.com/Christensen-Lab-Dartmouth/PyMethylProcess), and uses PyTorch to explore/make predictions on the methylation data.

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

**Install (Conda highly recommended, though Docker can be used, as well as nvidia-docker and Singularity for GPU usage, Docker image to come):**
* conda create -n methylnet python=3.6  
* source activate methylnet  
* See install instructions for PyMethylProcess at https://github.com/Christensen-Lab-Dartmouth/PyMethylProcess  
* conda install pytorch torchvision -c pytorch  
* pip install methylnet  
* Alternative docker install coming out soon.  

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

**Example of Running Pipeline:**  
See ./example_scripts for examples on how to explicitly run:
![MethylNetPresentationSpring2019Lab 001](https://user-images.githubusercontent.com/19698023/55677358-f12a9200-58b3-11e9-8aaf-50536d2afb00.jpeg)
![MethylNetPresentationSpring2019Lab 002](https://user-images.githubusercontent.com/19698023/55677359-f12a9200-58b3-11e9-8533-ad486ee7a0e7.jpeg)
![MethylNetPresentationSpring2019Lab 003](https://user-images.githubusercontent.com/19698023/55677360-f12a9200-58b3-11e9-8bdc-987bb9c0122e.jpeg)
![MethylNetPresentationSpring2019Lab 004](https://user-images.githubusercontent.com/19698023/55677361-f12a9200-58b3-11e9-8381-976ec02424f2.jpeg)
![MethylNetPresentationSpring2019Lab 005](https://user-images.githubusercontent.com/19698023/55677362-f12a9200-58b3-11e9-8095-cd25fbbb33c5.jpeg)
![MethylNetPresentationSpring2019Lab 006](https://user-images.githubusercontent.com/19698023/55677363-f12a9200-58b3-11e9-9acd-32e7752785fe.jpeg)
![MethylNetPresentationSpring2019Lab 007](https://user-images.githubusercontent.com/19698023/55677364-f12a9200-58b3-11e9-857d-b4ec88c08b78.jpeg)
![MethylNetPresentationSpring2019Lab 008](https://user-images.githubusercontent.com/19698023/55677365-f12a9200-58b3-11e9-95e1-e144f9c56287.jpeg)
![MethylNetPresentationSpring2019Lab 009](https://user-images.githubusercontent.com/19698023/55677366-f1c32880-58b3-11e9-9082-a7cc89f71a7d.jpeg)
