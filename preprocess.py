import rpy2.robjects as robjects
import rpy2.interactive as r
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
import os, subprocess
import click
import glob
import numpy as np, pandas as pd
from collections import Counter
#import impyute
from sklearn.impute import SimpleImputer
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler
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
        biocinstaller.biocLite(robjects.vectors.StrVector(["minfi","ENmix",
                                "minfiData","sva","GEOquery","geneplotter"]))

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
        tcga = importr("TCGAbiolinks")
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

    def download_geo(self, query, output_dir):
        """library(GEOquery)"""
        import GEOparse
        base=importr('base')
        geo = importr("GEOquery")
        geo.getGEOSuppFiles(query)
        robjects.r["untar"]("{0}/{0}_RAW.tar".format(query), exdir = "{}/idat".format(query))
        idatFiles = robjects.r('list.files("{}/idat", pattern = "idat.gz$", full = TRUE)'.format(query))
        robjects.r["sapply"](idatFiles, robjects.r["gunzip"], overwrite = True)
        subprocess.call('mv {}/idat/*.idat {}/'.format(query, output_dir),shell=True)
        # FIXME Table, dataTable import
        pandas2ri.ri2py(robjects.r['as'](robjects.r('phenoData')(robjects.r("getGEO('{}')[[1]]".format(query))),'data.frame')).to_csv('{}/{}_clinical_info.csv'.format(output_dir,query))# ,GSEMatrix = FALSE
        # geo_query="GSE109381"
        # geo = importr("GEOquery")
        # base=importr('base')
        # g=robjects.r("getGEO('{}')".format(geo_query))
        # dollar = base.__dict__["$"]
        # dollar(g, "phenoData")
        # robjects.r['as'](robjects.r('phenoData')(g),'data.frame')

class PreProcessPhenoData:
    def __init__(self, pheno_sheet, idat_dir, header_line=0): # source: geo, tcga, custom
        self.xlsx = True if pheno_sheet.endswith('.xlsx') or pheno_sheet.endswith('.xls') else False
        if self.xlsx:
            self.pheno_sheet = pd.read_excel(pheno_sheet,header=header_line)
        else:
            self.pheno_sheet = pd.read_csv(pheno_sheet, header=header_line)
        self.idat_dir = idat_dir

    def format_geo(self, disease_class_column="methylation class:ch1"):
        idats = glob.glob("{}/*.idat".format(self.idat_dir))
        idat_basenames = np.unique(np.vectorize(lambda x: '_'.join(x.split('/')[-1].split('_')[:3]))(idats))
        idat_geo_map = dict(zip(np.vectorize(lambda x: x.split('_')[0])(idat_basenames),np.array(idat_basenames)))
        self.pheno_sheet['Basename'] = self.pheno_sheet['geo_accession'].map(idat_geo_map).map(lambda x: self.idat_dir+x)
        self.pheno_sheet = self.pheno_sheet[self.pheno_sheet['Basename'].isin(idat_basenames)]
        self.pheno_sheet = self.pheno_sheet[['Basename','geo_accession',disease_class_column]].rename(columns={'geo_accession':'AccNum',disease_class_column:'disease'})


    def format_tcga(self, mapping_file="barcode_mapping.txt"):
        idats = glob.glob("{}/*.idat".format(self.idat_dir))
        barcode_mappings = np.loadtxt(mapping_file,dtype=str)
        barcode_mappings[:,1] = np.vectorize(lambda x: '-'.join(x.split('-')[:4]))(barcode_mappings[:,1])
        barcode_mappings = {v:k for k,v in dict(barcode_mappings.tolist()).items()}
        self.pheno_sheet['Basename'] = self.pheno_sheet['bcr_patient_barcode'].map(barcode_mappings)
        idat_basenames = np.unique(np.vectorize(lambda x: '_'.join(x.split('/')[-1].split('_')[:2]))(idats))
        self.pheno_sheet = self.pheno_sheet[self.pheno_sheet['Basename'].isin(idat_basenames)]
        self.pheno_sheet.loc[:,['Basename']] = self.pheno_sheet['Basename'].map(lambda x: self.idat_dir+x)
        self.pheno_sheet = self.pheno_sheet[['Basename', 'disease', 'tumor_stage', 'vital_status', 'age_at_diagnosis', 'gender', 'race', 'ethnicity']].rename(columns={'tumor_stage':'stage','vital_status':'vital','age_at_diagnosis':'age'})


    def format_custom(self, basename_col, disease_class_column, include_columns={}):
        self.pheno_sheet['Basename'] = self.pheno_sheet[basename_col].map(lambda x: self.idat_dir+x)
        self.pheno_sheet['disease'] = self.pheno_sheet[disease_class_column]
        self.pheno_sheet = self.pheno_sheet[np.unique(['Basename', 'disease']+include_columns.keys())].rename(columns=include_columns)

    def merge(self, other_formatted_sheet):
        self.pheno_sheet = self.pheno_sheet.merge(other_formatted_sheet,how='inner', on='Basename')

    def concat(self, other_formatted_sheet):
        self.pheno_sheet=pd.concat([self.pheno_sheet,other_formatted_sheet],join='inner').reset_index(drop=True)

    def export(self, output_sheet_name):
        self.pheno_sheet.to_csv(output_sheet_name)
        print("Please move all other sample sheets out of this directory.")

    def get_categorical_distribution(self, key):
        return Counter(self.pheno_sheet[key])

    def remove_diseases(self,exclude_disease_list):
        self.pheno_sheet = self.pheno_sheet[~self.pheno_sheet['disease'].isin(exclude_disease_list)]


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
            self.RGset.slots["pData"] = robjects.r('pData')(robjects.r("getGEO('{}')[[1]]".format(query)))
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

class ImputerObject:
    def __init__(self, solver, method, opts={}):

        imputers = {'fancyimpute':dict(KNN=KNN(k=opts['k']),MICE=IterativeImputer(),BiScaler=BiScaler(),Soft=SoftImpute()),
                    'impyute':dict(),
                    'simple':dict(Mean=SimpleImputer(strategy='mean'),Zero=SimpleImputer(strategy='constant'))}
        try:
            self.imputer = imputers[solver][method]
        except:
            print('{} {} not a valid combination.\nValid combinations:{}'.format(
                solver, method, '\n'.join('{}:{}'.format(solver,','.join(imputers[solver].keys())) for solver in imputers)))
            exit()

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
@click.option('-g', '--geo_query', default='', help='GEO study to query.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./geo_idats/', help='Output directory for exported idats.', type=click.Path(exists=False), show_default=True)
def download_geo(geo_query,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    downloader = TCGADownloader()
    downloader.download_geo(geo_query,output_dir)

@preprocess.command()
@click.option('-is', '--input_sample_sheet', default='./tcga_idats/clinical_info.csv', help='Clinical information downloaded from tcga/geo/custom.', type=click.Path(exists=False), show_default=True)
@click.option('-s', '--source_type', default='tcga', help='Source type of data.', type=click.Choice(['tcga','geo','custom']), show_default=True)
@click.option('-i', '--idat_dir', default='./tcga_idats/', help='Idat directory.', type=click.Path(exists=False), show_default=True)
@click.option('-os', '--output_sample_sheet', default='./tcga_idats/minfiSheet.csv', help='CSV for minfi input.', type=click.Path(exists=False), show_default=True)
@click.option('-m', '--mapping_file', default='./barcode_mapping.txt', help='Mapping file from uuid to TCGA barcode.', type=click.Path(exists=False), show_default=True)
@click.option('-h', '--header_line', default=0, help='Line to begin reading csv/xlsx.', show_default=True)
@click.option('-d', '--disease_class_column', default="methylation class:ch1", help='Disease classification column, for custom and geo datasets.', type=click.Path(exists=False), show_default=True)
@click.option('-b', '--basename_col', default="Sentrix ID (.idat)", help='Basename classification column, for custom datasets.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--include_columns_file', default="", help='Custom columns file containing columns to keep, separated by \\n. Add a tab for each line if you wish to rename columns: original_name \\t new_column_name', type=click.Path(exists=False), show_default=True)
def create_sample_sheet(input_sample_sheet, source_type, idat_dir, output_sample_sheet, mapping_file, header_line, disease_class_column, basename_col, include_columns_file):
    os.makedirs(output_dir, exist_ok=True)
    pheno_sheet = PreProcessPhenoData(input_sample_sheet, idat_dir, header_line=0 if source_type is not 'custom' else header_line)
    if source_type == 'tcga':
        pheno_sheet.format_tcga(mapping_file)
    elif source_type == 'geo':
        pheno_sheet.format_geo(disease_class_column)
    else:
        if include_columns_file:
            include_columns=dict(np.loadtxt(include_columns_file,dtype=str,delimiter='\t').tolist())
        else:
            include_columns={}
        pheno_sheet.format_custom(basename_col, disease_class_column, include_columns)
    pheno_sheet.export(output_sheet_name)
    print("Please remove {} from {}, if it exists in that directory.".format(input_sample_sheet, idat_dir))

@preprocess.command()
@click.option('-s1', '--sample_sheet1', default='./tcga_idats/clinical_info1.csv', help='Clinical information downloaded from tcga/geo/custom, formatted using create_sample_sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-s2', '--sample_sheet2', default='./tcga_idats/clinical_info2.csv', help='Clinical information downloaded from tcga/geo/custom, formatted using create_sample_sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-os', '--output_sample_sheet', default='./tcga_idats/minfiSheet.csv', help='CSV for minfi input.', type=click.Path(exists=False), show_default=True)
def merge_sample_sheets(sample_sheet1, sample_sheet2, output_sample_sheet):
    s1 = PreProcessPhenoData(sample_sheet1, idat_dir='', header_line=0)
    s2 = PreProcessPhenoData(sample_sheet2, idat_dir='', header_line=0)
    s1.merge(s2)
    s1.export(output_sample_sheet)

@preprocess.command()
@click.option('-s1', '--sample_sheet1', default='./tcga_idats/clinical_info1.csv', help='Clinical information downloaded from tcga/geo/custom, formatted using create_sample_sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-s2', '--sample_sheet2', default='./tcga_idats/clinical_info2.csv', help='Clinical information downloaded from tcga/geo/custom, formatted using create_sample_sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-os', '--output_sample_sheet', default='./tcga_idats/minfiSheet.csv', help='CSV for minfi input.', type=click.Path(exists=False), show_default=True)
def concat_sample_sheets(sample_sheet1, sample_sheet2, output_sample_sheet):
    s1 = PreProcessPhenoData(sample_sheet1, idat_dir='', header_line=0)
    s2 = PreProcessPhenoData(sample_sheet2, idat_dir='', header_line=0)
    s1.concat(s2)
    s1.export(output_sample_sheet)

@preprocess.command()
@click.option('-is', '--formatted_sample_sheet', default='./tcga_idats/clinical_info.csv', help='Clinical information downloaded from tcga/geo/custom, formatted using create_sample_sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-k', '--key', default='disease', help='Column of csv to print statistics for.', type=click.Path(exists=False), show_default=True)
def get_categorical_distribution(formatted_sample_sheet,key):
    print('\n'.join('{}:{}'.format(k,v) for k,v in PreProcessPhenoData(formatted_sample_sheet, idat_dir='', header_line=0).get_categorical_distribution(key)))

@preprocess.command()
@click.option('-is', '--formatted_sample_sheet', default='./tcga_idats/clinical_info.csv', help='Clinical information downloaded from tcga/geo/custom, formatted using create_sample_sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-e', '--exclude_disease_list', default='', help='List of conditions to exclude, from disease column.', type=click.Path(exists=False), show_default=True)
@click.option('-os', '--output_sample_sheet', default='./tcga_idats/minfiSheet.csv', help='CSV for minfi input.', type=click.Path(exists=False), show_default=True)
def remove_diseases(formatted_sample_sheet, exclude_disease_list, output_sheet_name):
    exclude_disease_list = exclude_disease_list.split(',')
    pData = PreProcessPhenoData(formatted_sample_sheet, idat_dir='', header_line=0)
    pData.remove_diseases(exclude_disease_list)
    pData.export(output_sheet_name)
    print("Please remove {} from idat directory, if it exists in that directory.".format(formatted_sample_sheet))


### TODO: Wrap a class around following functions ###

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
@click.option('-i', '--idat_dir', default='./tcga_idats/', help='Idat directory if one sample sheet, alternatively can be your phenotype sample sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-g', '--geo_query', default='', help='GEO study to query, do not use if already created geo sample sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_dir', default='./preprocess_outputs/', help='Output directory for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-d', '--split_by_subtype', is_flag=True, help='If using formatted sample sheet csv, split by subtype and perform preprocessing. Will need to combine later.')
def plot_qc(idat_dir, geo_query, output_dir, split_by_subtype):
    os.makedirs(output_dir, exist_ok=True)
    if idat_dir.endswith('.csv') and split_by_subtype:
        pheno=pd.read_csv(idat_dir)
        for name, group in pheno.groupby('disease'):
            new_sheet = idat_dir.replace('.csv','_{}.csv'.format(name))
            new_out_dir = '{}/{}/'.format(output_dir,name)
            group.to_csv(new_sheet)
            os.makedirs(new_out_dir, exist_ok=True)
            preprocesser = PreProcessIDAT(new_sheet)
            preprocesser.load_idats(geo_query='')
            preprocesser.plot_original_qc(new_out_dir)
    else:
        preprocesser = PreProcessIDAT(idat_dir)
        preprocesser.load_idats(geo_query)
        preprocesser.plot_original_qc(output_dir)

@preprocess.command()
@click.option('-i', '--idat_dir', default='./tcga_idats/', help='Idat directory if one sample sheet, alternatively can be your phenotype sample sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-g', '--geo_query', default='', help='GEO study to query, do not use if already created geo sample sheet.', type=click.Path(exists=False), show_default=True)
@click.option('-n', '--n_cores', default=6, help='Number cores to use for preprocessing.', show_default=True)
@click.option('-o', '--output_dir', default='./preprocess_outputs/', help='Output directory for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-d', '--split_by_subtype', is_flag=True, help='If using formatted sample sheet csv, split by subtype and perform preprocessing. Will need to combine later.')
def preprocess_pipeline(idat_dir, geo_query, n_cores, output_dir, split_by_subtype):
    os.makedirs(output_dir, exist_ok=True)
    if idat_dir.endswith('.csv') and split_by_subtype:
        pheno=pd.read_csv(idat_dir)
        for name, group in pheno.groupby('disease'):
            new_sheet = idat_dir.replace('.csv','_{}.csv'.format(name))
            new_out_dir = '{}/{}/'.format(output_dir,name)
            group.to_csv(new_sheet)
            os.makedirs(new_out_dir, exist_ok=True)
            preprocesser = PreProcessIDAT(new_sheet)
            preprocesser.preprocess(geo_query='', n_cores=n_cores)
            preprocesser.output_pheno_beta(new_out_dir)
    else:
        preprocesser = PreProcessIDAT(idat_dir)
        preprocesser.preprocess(geo_query, n_cores)
        preprocesser.output_pheno_beta(output_dir)

@preprocess.command()
@click.option('-d', '--split_by_subtype', is_flag=True, help='Imputes CpGs by subtype before combining again.')
@click.option('-m', '--method', default='knn', help='Method of imputation.', type=click.Choice(['KNN', 'Mean', 'Zero', 'MICE', 'BiScaler', 'Soft', 'random', 'DeepCpG', 'DAPL']), show_default=True)
@click.option('-s', '--solver', default='fancyimpute', help='Imputation library.', type=click.Choice(['fancyimpute', 'impyute', 'simple']), show_default=True)
@click.option('-k', '--n_neighbors', default=5, help='Number neioghbors for imputation if using KNN.', show_default=True)

def imputation_pipeline(split_by_subtype=True, method='knn', solver='fancyimpute', n_neighbors=5): # wrap a class around this
    """Imputation of subtype or no subtype using """
    if method in ['DeepCpG', 'DAPL', 'EM']:
        print('Method {} coming soon...'.format(method))
    elif solver in ['impyute']:
        print('Impyute coming soon...')
    else:
        imputer = ImputerObject(solver, method, dict(k=n_neighbors))

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
