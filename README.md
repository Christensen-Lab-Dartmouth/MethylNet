# MethylNet

Deep Learning with Methylation

MethylNet is a command line tool and python library that provides classes to handle deep learning tasks for methylation data. It wraps common R-bioconductor functions for the preliminary analysis of methylation arrays, and uses PyTorch to explore/make predictions on the methylation data.

What MethylNet can do:  
1. Preprocessing of IDATs (wraps R into python using rpy2):  
                * Install bioconductor packages (TCGAbiolinks, minfi, enmix, etc..)  
                * Download TCGA and GEO data and clinical information  
                * Preprocessing pipeline via enmix and minfi  
                * Imputation using various methods (knn, MICE, mean, zero, etc.) and MAD (Mean Absolute Deviation) filtering of CpGs using fancyimpute and pandas  
                * Storage of methylation beta value array and phenotype data into sqlite3 databases to query, embed, explore (QEE)  
2. Deep Learning Tools via PyTorch:  
                * Discover latent space of methylation data using Tybalt-like model and Convolutional Variational Autoencoders  
                * UMAP/TSNE embedding of latent space  
                * Semantic arithmetic of latent space  
                * Predictions on latent space  
                * Mapback functions to discover relevant biological features  
                * Transfer learning of VAE to Convolutional Neural Net (CNN) (directly using encoder network or learned using DeepLift)  
                * Using CNN or VAE+MLP to make predictions  
                * Model Interpretation using SHAP, LIME, and Grad-CAM  
                * And more!  
3. Planned Applications  
                * Brain Cancer Classifier and Semantic exploration (pathway and context analysis, etc)  
                * Pancancer latent space exploration (pathway and context analysis, etc)  
                * Hybrid cell type deconvolution method with reference based approach  
                * Inclusion in cross-media retrieval and integration with histopathology images for stronger and interchangeable predictions  
                * Possible: Learning hierarchical semantic representations of various biological ontological networks (GO, KEGG, etc.)  
                * And more!
4. Misc.  
                * Job deployment system using Toil (PBS/Torque) for high performance computing

Getting Started:  
0. Small test cases: GSE64950 (removed because Mars study) , GSE104376  
                * python preprocess.py download_geo -g GSE104376 -o test_idats/  
                * python preprocess.py create_sample_sheet -s geo -i test_idats/ -os ./test_idats/geo_minfi.csv -is test_idats/GSE104376_clinical_info.csv -d "gender:ch1" # gender is "disease" as mock test  
                * python preprocess.py plot_qc -i ./test_idats/  
                * python preprocess.py preprocess_pipeline -i ./test_idats/ -n 30  
                * python preprocess.py imputation_pipeline -i ./preprocess_outputs/methyl_array.pkl
                * python preprocess.py print_na_rate -i ./imputed_outputs/methyl_array.pkl
                * python preprocess.py mad_filter -n 200000
                * python preprocess.py pkl_to_csv # output in final_preprocessed
                * python visualizations.py transform_plot
1. Log into server ssh xxxxx@discovery7.hpcc.dartmouth.edu  
                * ssh x01  
                * module load python/3-Anaconda#module load python/3.6-Miniconda  
                * module load cuda  
2. Download or load anaconda environment:  
                * install conda environment from https://anaconda.org/jlevy44/methylnet # note imcomplete number of packages, may consider docker container  
                * source activate methylnet  
3. Download data:  
                * python preprocess.py download_clinical -h    
                * python preprocess.py download_geo -h    
                * python preprocess.py download_geo -g GSE109381 -o geo_idats/  
                * python preprocess.py download_tcga -h    
4. Format clinical data for minfi processing:   
                * Brain Cancer example: GSE109381 (train and test sheets from supplementals)  
                * python preprocess.py create_sample_sheet -h    
                * make custom include_columns_file <- tab delimited file of header of csv file header csv column name\\t desired name  
                * python preprocess.py create_sample_sheet -s custom -is geo_idats/train.xlsx -l 1 -i geo_idats/ -os ./geo_idats/minfiSheet.csv -d "Pathological Diagnosis (WHO 2016)" -b "Sentrix ID (.idat)" -c include_col.txt  
                * python preprocess.py create_sample_sheet -s custom -is geo_idats/test.xlsx -l 1 -i geo_idats/ -os ./geo_idats/minfiSheet2.csv -d "Pathological diagnosis (WHO 2016) prior to methylation classification" -b "Sentrix ID (.idat)" -c include_col2.txt  
                * python preprocess.py concat_sample_sheets -s1 ./geo_idats/minfiSheet.csv -s2 ./geo_idats/minfiSheet2.csv -os ./geo_idats/geo_concat.csv  
                * python preprocess.py create_sample_sheet -s geo -i geo_idats/ -os ./geo_idats/geo_minfi.csv -is geo_idats/GSE109381_clinical_info.csv  
                * python preprocess.py merge_sample_sheets -s1 ./geo_idats/geo_minfi.csv -s2 ./geo_idats/geo_concat.csv -os ./geo_idats/geo_merged.csv  
                * python preprocess.py get_categorical_distribution -is ./geo_idats/geo_merged.csv  
                * python preprocess.py get_categorical_distribution -is ./geo_idats/geo_merged.csv -d  
                * python preprocess.py get_categorical_distribution -is ./geo_idats/geo_merged.csv -d | awk -F':' '{sum+=$2} END {print sum}'  
                * Prints out total number of cases/controls 3905  
5. Preprocessing pipeline with minfi and enmix:  
                * python preprocess.py plot_qc -i ./geo_idats/  
                * python preprocess.py preprocess_pipeline -i ./geo_idats/  

TODO:
* Test Preprocessing tools   
* Test VAE  
* More  
