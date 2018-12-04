# MethylNet

Deep Learning with Methylation

MethylNet is a command line tool and python library that provides classes to handle deep learning tasks for methylation data. It wraps common R-bioconductor functions for the preliminary analysis of methylation arrays, and uses PyTorch to explore/make predictions on the methylation data.

What MethylNet can do:
* Preprocessing of IDATs (wraps R into python using rpy2):
        * Install bioconductor packages (TCGAbiolinks, minfi, enmix, etc..)
        * Download TCGA and GEO data and clinical information
        * Preprocessing pipeline via enmix and minfi
        * Imputation using various methods (knn, MICE, mean, zero, etc.) and MAD (Mean Absolute Deviation) filtering of CpGs using fancyimpute and pandas
        * Storage of methylation beta value array and phenotype data into sqlite3 databases to query, embed, explore (QEE)
* Deep Learning Tools via PyTorch:
        * Discover latent space of methylation data using Tybalt-like model and Convolutional Variational Autoencoders
        * UMAP/TSNE embedding of latent space
        * Semantic arithmetic of latent space
        * Predictions on latent space
        * Mapback functions to discover relevant biological features
        * Transfer learning of VAE to Convolutional Neural Net (CNN) (directly using encoder network or learned using DeepLift)
        * Using CNN or VAE+MLP to make predictions
        * Model Interpretation using SHAP, LIME, and Grad-CAM
        * And more!
* Planned Applications
        * Brain Cancer Classifier and Semantic exploration (pathway and context analysis, etc)
        * Pancancer latent space exploration (pathway and context analysis, etc)
        * Hybrid cell type deconvolution method with reference based approach
        * Inclusion in cross-media retrieval and integration with histopathology images for stronger and interchangeable predictions
        * Possible: Learning hierarchical semantic representations of various biological ontological networks (GO, KEGG, etc.)
        * And more!

TODO:
* Test Preprocessing tools
* Test VAE
