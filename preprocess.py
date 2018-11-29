import rpy2.robjects as robjects
import rpy2.interactive as r
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
import os
import click
import glob
import numpy as np, pandas as pd

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def preprocess():
    pass

class PackageInstaller:
    def __init__(self):
        pass

    def install_bioconductor(self):
        base = importr('base')
        base.source("http://www.bioconductor.org/biocLite.R")

    def install_tcga_biolinks(self):
        #robjects.r('source("https://bioconductor.org/biocLite.R")\n')
        biocinstaller = importr("BiocInstaller")
        biocinstaller.biocLite("TCGAbiolinks")

    def install_minfi(self):
        biocinstaller = importr("BiocInstaller")
        biocinstaller.biocLite("minfi")
        biocinstaller.biocLite("ENmix")


class TCGADownloader:
    def __init__(self):
        pass

        """utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        packnames=('TCGAbiolinks',)
        from rpy2.robjects.vectors import StrVector
        names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))"""

    def download_tcga(self, output_dir):
        import rpy2.interactive as r
        import rpy2.interactive.packages
        tcga = r.packages.importr("TCGAbiolinks")
        print(tcga)
        robjects.r("""
                   library(TCGAbiolinks)
                   projects <- TCGAbiolinks:::getGDCprojects()$project_id
                   projects <- projects[grepl('^TCGA',projects,perl=T)]
                   match.file.cases.all <- NULL
                   for(proj in projects){
                        print(proj)
                        query <- GDCquery(project = proj,
                                          data.category = "Raw microarray data",
                                          data.type = "Raw intensities",
                                          experimental.strategy = "Methylation array",
                                          legacy = TRUE,
                                          file.type = ".idat",
                                          platform = "Illumina Human Methylation 450")
                        match.file.cases <- getResults(query,cols=c("cases","file_name"))
                        match.file.cases$project <- proj
                        match.file.cases.all <- rbind(match.file.cases.all,match.file.cases)
                        tryCatch(GDCdownload(query, method = "api", files.per.chunk = 20),
                                 error = function(e) GDCdownload(query, method = "client"))
                    }
                    # This will create a map between idat file name, cases (barcode) and project
                    readr::write_tsv(match.file.cases.all, path =  "idat_filename_case.txt")
                    # code to move all files to local folder
                    for(file in dir(".",pattern = ".idat", recursive = T)){
                        TCGAbiolinks:::move(file,file.path('%s',basename(file)))
                    }
                   """%output_dir)

    def download_clinical(self, output_dir):
        robjects.r("""
                   library(TCGAbiolinks)
                   library(data.table)
                   projects <- TCGAbiolinks:::getGDCprojects()$project_id
                   projects <- projects[grepl('^TCGA',projects,perl=T)]
                   match.file.cases.all <- NULL
                   data <- list()
                   for(n in 1:length(projects)){
                        proj <- projects[n]
                        print(proj)
                        clin.query <- GDCquery_clinic(project = proj,
                                          type='clinical', save.csv=F)
                        #tryCatch(GDCdownload(clin.query, method='api'),
                        #    error = function(e) {GDCdownload(clin.query, method='client')})
                        #tryCatch(clinical.patient <- GDCprepare_clinic(clin.query, clinical.info = "patient"),
                        #    error = function(e) {0})
                        data[[length(data)+1]] = clin.query
                    }
                    df <- rbindlist(data)
                    write.csv(df, file=file.path('%s','clinical_info.csv'))
                   """%output_dir)

    def create_sample_sheet(self, input_sample_sheet, idat_dir, output_sample_sheet, mapping_file="barcode_mapping.txt"):
        idats = glob.glob("{}/*.idat".format(idat_dir))
        barcode_mappings = np.loadtxt(mapping_file,dtype=str)
        barcode_mappings[:,1] = np.vectorize(lambda x: '-'.join(x.split('-')[:4]))(barcode_mappings[:,1])
        barcode_mappings = {v:k for k,v in dict(barcode_mappings.tolist()).items()}
        input_df = pd.read_csv(input_sample_sheet)
        input_df['Basename'] = input_df['bcr_patient_barcode'].map(barcode_mappings)
        idat_basenames = np.unique(np.vectorize(lambda x: '_'.join(x.split('_')[:2]))(idats))
        output_df = input_df[input_df['Basename'].isin(idat_basenames)]
        output_df.loc[:,['Basename']] = output_df['Basename'].map(lambda x: idat_dir+x)
        output_df.to_csv(output_sample_sheet)
        #tcga_files = glob.glob(head_tcga_dir+'/*')
        #pd.read_csv('clinical_info.csv')

class PreProcessIDAT:
    def __init__(self, idat_dir):
        self.idat_dir = idat_dir # can establish cases and controls

    def load_idats(self):
        robjects.r("""targets <- read.450k.sheet({})
            sub({}, "", targets$Basename)
            RGset <- read.450k.exp(base = {}, targets = targets)""".format(self.idat_dir))
        self.RGset = robjects.globalenv['RGset']

    def print_idats(self):
        #print(RGset)
        #x=1
        #robjects.r.assign('x',x) # example python object to r
        #robjects.globalenv['RGset']
        robjects.r("print(RGset)")

#### COMMANDS ####

## Install ##
@preprocess.command()
def install_bioconductor():
    installer = PackageInstaller()
    installer.install_bioconductor()

@preprocess.command()
def install_minfi():
    installer = PackageInstaller()
    installer.install_minfi()

@preprocess.command()
def install_tcga_biolinks():
    installer = PackageInstaller()
    installer.install_tcga_biolinks()

## Download ##

@preprocess.command()
@click.option('-o', '--output_dir', default='./tcga_idats/', help='Output directory for exported idats.', type=click.Path(exists=False), show_default=True)
def download_tcga(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    downloader = TCGADownloader()
    downloader.download_tcga(output_dir)

@preprocess.command()
@click.option('-o', '--output_dir', default='./tcga_idats/', help='Output directory for exported idats.', type=click.Path(exists=False), show_default=True)
def download_clinical(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    downloader = TCGADownloader()
    downloader.download_clinical(output_dir)

@preprocess.command()
@click.option('-is', '--input_sample_sheet', default='./tcga_idats/clinical_info.csv', help='Clinical information downloaded from tcga.', type=click.Path(exists=False), show_default=True)
@click.option('-i', '--idat_dir', default='./tcga_idats/', help='Idat directory.', type=click.Path(exists=False), show_default=True)
@click.option('-os', '--output_sample_sheet', default='./tcga_idats/minfiSheet.csv', help='CSV for minfi input.', type=click.Path(exists=False), show_default=True)
@click.option('-m', '--mapping_file', default='./barcode_mapping.txt', help='Mapping file from uuid to TCGA barcode.', type=click.Path(exists=False), show_default=True)
def create_sample_sheet(input_sample_sheet, idat_dir, output_sample_sheet, mapping_file):
    os.makedirs(output_dir, exist_ok=True)
    downloader = TCGADownloader()
    downloader.create_sample_sheet(input_sample_sheet, idat_dir, output_sample_sheet, mapping_file)
    print("Please remove {} from {}, if it exists in that directory.".format(input_sample_sheet, idat_dir))

## preprocess ##

@preprocess.command()
@click.option('-i', '--idat_dir', default='./tcga_idats/', help='Idat directory.', type=click.Path(exists=False), show_default=True)
def load_idats(idat_dir):
    preprocesser = PreProcessIDAT(idat_dir)
    preprocesser.load_idats()
    preprocesser.print_idats()

#################

if __name__ == '__main__':
    preprocess()
