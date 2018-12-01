import rpy2.robjects as robjects
import rpy2.interactive as r
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
import os, subprocess
import click
import glob
import numpy as np, pandas as pd
from rpy2.robjects import pandas2ri
pandas2ri.activate()

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

    def install_minfi_others(self):
        biocinstaller = importr("BiocInstaller")
        biocinstaller.biocLite("minfi")
        biocinstaller.biocLite("ENmix")
        biocinstaller.biocLite("minfiData")
        biocinstaller.biocLite("sva")
        biocinstaller.biocLite("GEOquery")
        biocinstaller.biocLite("geneplotter")




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

    def download_geo(self, query, output_dir):
        """library(GEOquery)"""
        base=importr('base')
        geo = importr("GEOquery")
        geo.getGEOSuppFiles(query)
        geo.untar("{}/{}_RAW.tar".format(query), exdir = "{}/idat".format(query))
        idatFiles = robjects.r('list.files("{}/idat", pattern = "idat.gz$", full = TRUE)'.format(query))
        base.sapply(idatFiles, base.gunzip, overwrite = True)
        subprocess.call('mv {}/idat/*.idat {}/'.format(query, output_dir),shell=True)
        # FIXME Table, dataTable import
        pandas2ri.ri2py(base.as_data_frame_matrix(robjects.r("Table(dataTable(getGEO('{}')[[1]]))".format(geo_query)))).to_csv('{}/{}_clinical_info.csv'.format(output_dir,query))


class PreProcessIDAT:
    # https://kasperdanielhansen.github.io/genbioconductor/html/minfi.html
    # https://www.bioconductor.org/help/course-materials/2015/BioC2015/methylation450k.html#dependencies
    def __init__(self, idat_dir):
        self.idat_dir = idat_dir # can establish cases and controls
        self.minfi = importr('minfi')
        self.enmix = importr("ENmix")

    def load_idats(self, geo_query=''): # maybe have a larger parent class that subsets idats by subtype, then preprocess each subtype and combine the dataframes
        targets = self.minfi.read_450k_sheet(self.idat_dir)
        self.RGset = self.minfi.read_450k_exp(targets=targets, extended=True)
        if geo_query:
            geo = importr('GEOquery')
            self.RGset.slots["pData"] = self.minfi.pData(robjects.r("getGEO('{}')[[1]]".format(geo_query)))
            print(self.RGset.slots["pData"])
        #robjects.r("""targets <- read.450k.sheet({})
        #    sub({}, "", targets$Basename)
        #    RGset <- read.450k.exp(base = {}, targets = targets)""".format(self.idat_dir))
        #self.RGset = robjects.globalenv['RGset']
        return self.RGset

    def preprocessRAW(self):
        self.MSet = self.minfi.preprocessRAW(self.RGset)
        return self.MSet

    def preprocessENmix(self, n_cores=6):
        self.qcinfo = self.enmix.QCinfo(self.RGset, detPthre=1e-7)
        self.MSet = self.enmix.preprocessENmix(self.RGset, QCinfo=self.qcinfo, nCores=n_cores)
        self.MSet = self.enmix.QCfilter(self.MSet,qcinfo=self.qcinfo,outlier=True)
        return self.MSet

    def return_beta(self):
        self.RSet = self.minfi.ratioConvert(self.MSet, what = "both", keepCN = True)
        return self.RSet

    def get_beta(self):
        self.beta = self.minfi.getBeta(self.RSet, "Illumina")
        return self.beta

    def filter_beta(self):
        self.beta_final=self.enmix.rm_outlier(self.beta,qcscore=self.qcinfo)
        return self.beta_final

    def plot_qc_metrics(self, output_dir):
        self.enmix.plotCtrl(self.RGset)
        grdevice = importr("grDevices")
        geneplotter = importr("geneplotter")
        base = importr('base')
        anno=self.minfi.getAnnotation(self.RGset)
        self.enmix.multifreqpoly(self.get_meth()+self.get_unmeth(), xlab="Total intensity")
        beta_py = pandas2ri.ri2py(self.beta)
        anno_py = pandas2ri.ri2py(anno)
        beta1=pandas2ri.py2ri(beta_py[anno_py["Type"]=="I"])
        beta2=pandas2ri.py2ri(beta_py[anno_py["Type"]=="II"])
        grdevice.jpeg(output_dir+'/dist.jpg',height=900,width=600)
        base.par(mfrow=robjects.vectors.IntVector([3,2]))
        self.enmix.multidensity(self.beta, main="Multidensity")
        self.enmix.multifreqpoly(self.beta, xlab="Beta value")
        self.enmix.multidensity(beta1, main="Multidensity: Infinium I")
        self.enmix.multifreqpoly(beta1, main="Multidensity: Infinium I", xlab="Beta value")
        self.enmix.multidensity(beta2, main="Multidensity: Infinium II")
        self.enmix.multifreqpoly(beta2, main="Multidensity: Infinium II", xlab="Beta value")
        grdevice.dev_off()
        self.minfi.qcReport(self.RGset, pdf = "{}/qcReport.pdf".format(output_dir))  # sampNames = pheno$X_SAMPLE_ID, sampGroups = pheno$sample.type,
        self.minfi.mdsPlot(self.RGset)#, sampNames = pheno$X_SAMPLE_ID, sampGroups = pheno$sample.type)
        self.minfi.densityPlot(self.RGset, main='Beta', xlab='Beta')#, sampGroups = pheno$sample.type, main = "Beta", xlab = "Beta")

    def get_meth(self):
        return self.minfi.getMeth(self.MSet)

    def get_unmeth(self):
        return self.minfi.getUnmeth(self.MSet)

    def extract_pheno_data(self, methylset=False):
        self.pheno = self.minfi.pData(self.Mset) if methylset else self.minfi.pData(self.RGset)
        return self.pheno

    def extract_manifest(self):
        self.manifest = self.minfi.getManifest(self.RGset)
        return self.manifest

    def preprocess(self, geo_query='', n_cores=6):
        self.load_idats(geo_query)
        self.preprocessENmix(n_cores)
        self.return_beta()
        self.get_beta()
        self.filter_beta()
        self.extract_pheno_data(methylset=True)
        return self.pheno, self.beta_final

    def plot_original_qc(self, output_dir):
        self.preprocessRAW()
        self.return_beta()
        self.get_beta()
        self.plot_qc_metrics(output_dir)

    def output_pheno_beta(self, output_dir):
        pandas2ri.ri2py(self.pheno).to_csv('{}/pheno.csv'.format(output_dir))
        pandas2ri.ri2py(self.beta_final).to_csv('{}/beta.csv'.format(output_dir))

#### COMMANDS ####

## Install ##
@preprocess.command()
def install_bioconductor():
    installer = PackageInstaller()
    installer.install_bioconductor()

@preprocess.command()
def install_minfi_others():
    installer = PackageInstaller()
    installer.install_minfi_others()

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

### Wrap a class around following functions ###
def make_standard_sheet():
    """Geo and TCGAbiolinks standardize phenotype data"""
    pass

def print_case_controls():
    """Print number of case and controls for subtypes"""
    pass

def remove_controls():
    """Remove controls for study"""
    pass

def remove_low_sample_number():
    """Remove cases for study with low sample number"""
    pass

## preprocess ##

@preprocess.command() # update
@click.option('-i', '--idat_dir', default='./tcga_idats/', help='Idat directory.', type=click.Path(exists=False), show_default=True)
def plot_qc(idat_dir, geo_query, n_cores, output_dir):
    preprocesser = PreProcessIDAT(idat_dir)
    preprocesser.load_idats(geo_query)
    preprocesser.plot_original_qc(output_dir)

@preprocess.command()
@click.option('-i', '--idat_dir', default='./tcga_idats/', help='Idat directory.', type=click.Path(exists=False), show_default=True)
def preprocess_pipeline(idat_dir, geo_query, n_cores, output_dir):
    preprocesser = PreProcessIDAT(idat_dir)
    preprocesser.preprocess(geo_query, n_cores)
    preprocesser.output_pheno_beta(output_dir)

def imputation_pipeline(subtype_split=True, method='knn'): # wrap a class around this
    """Imputation of subtype or no subtype using """
    pass

def remove_MAD_threshold():
    """Filter CpGs below MAD threshold"""
    pass

## Build methylation class with above features ##

## Build MethylNet (sklearn interface) and Pickle ##
# methylnet class features various dataloaders, data augmentation methods, different types variational autoencoders (load from other), with customizable architecture, etc, skorch?

# use another script for this: https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66

#################

if __name__ == '__main__':
    preprocess()
