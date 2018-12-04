# MethylNet

Deep Learning with Methylation

What MethylNet can do:
* Preprocessing of IDATs (wraps R into python using rpy2):
        * Install bioconductor packages (TCGAbiolinks, minfi, enmix, etc..)
        * Download TCGA and GEO data and clinical information
        * Preprocessing pipeline via enmix and minfi
        * Imputation using various methods (knn, MICE, mean, zero, etc.) and MAD (Mean Absolute Deviation) filtering of CpGs using fancyimpute and pandas
        * Storage of methylation beta value array and phenotype data into sqlite3 databases to query, embed, explore (QEE)
* Deep Learning Tools via PyTorch:
        * Discover latent space of methylation data using Tybalt and Convolutional Variational Autoencoders
        * UMAP/TSNE embedding of latent space
        * Semantic arithmetic of latent space
        * Predictions on latent space
        * Mapback functions to discover relevant biological features
        * Transfer learning of VAE to Convolutional Neural Net (CNN) (directly using encoder network or learned using DeepLift)
        * Using CNN or VAE+MLP to make predictions
        * And more!
* Planned Applications
        * Brain Cancer Classifier and Semantic exploration (pathway and context analysis, etc)
        * Pancancer latent space exploration (pathway and context analysis, etc)
        * Hybrid cell type deconvolution method with reference based approach
        * Inclusion in cross-media retrieval and integration with histopathology images for stronger and interchangeable predictions
        * Possible: Learning hierarchical semantic representations of various biological ontological networks (GO, KEGG, etc.)
        * And more!
