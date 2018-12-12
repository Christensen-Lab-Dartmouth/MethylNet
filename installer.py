from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import click


CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def install():
    pass

class PackageInstaller:
    def __init__(self):
        pass

    def install_bioconductor(self):
        #robjects.r['install.packages']("BiocInstaller",repos="http://bioconductor.org/packages/3.7/bioc")
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

    def install_custom(self, custom, manager):
        if not manager:
            biocinstaller = importr("BiocInstaller")
            biocinstaller.biocLite(robjects.vectors.StrVector(custom),suppressUpdates=True)
        else:
            biocinstaller = importr("BiocManager")
            for c in custom:
                if '=' in c:
                    pkg,version= tuple(c.split('='))
                    biocinstaller.install(pkg,ask=False,version=version)
                else:
                    biocinstaller.install(c,ask=False)

    def install_devtools(self):
        subprocess.call('conda install -y -c r r-cairo=1.5_9 r-devtools=1.13.6',shell=True)
        robjects.r('install.packages')('devtools')

    def install_r_packages(self, custom):
        robjects.r["options"](repos=robjects.r('structure(c(CRAN="http://cran.wustl.edu/"))'))
        robjects.r('install.packages')(robjects.vectors.StrVector(custom))

    def install_meffil(self):
        #base = importr('base')
        #base.source("http://www.bioconductor.org/biocLite.R")
        #biocinstaller = importr("BiocInstaller")
        remotes=importr('remotes')
        remotes.install_github('perishky/meffil')
        #devtools=importr('devtools')
        #devtools.install_git("https://github.com/perishky/meffil.git")


## Install ##
@install.command()
def install_bioconductor():
    """Installs bioconductor."""
    installer = PackageInstaller()
    installer.install_bioconductor()

@install.command()
@click.option('-p', '--package', multiple=True, default=['ENmix'], help='Custom packages.', type=click.Path(exists=False), show_default=True)
@click.option('-m', '--manager', is_flag=True, help='Use BiocManager (recommended).')
def install_custom(package,manager):
    """Installs bioconductor packages."""
    installer = PackageInstaller()
    installer.install_custom(package,manager)

@install.command()
@click.option('-p', '--package', multiple=True, default=[''], help='Custom packages.', type=click.Path(exists=False), show_default=True)
def install_r_packages(package):
    """Installs r packages."""
    installer = PackageInstaller()
    installer.install_r_packages(package)

@install.command()
def install_minfi_others():
    """Installs minfi and other dependencies."""
    installer = PackageInstaller()
    installer.install_minfi_others()

@install.command()
def install_tcga_biolinks():
    """Installs tcga biolinks."""
    installer = PackageInstaller()
    installer.install_tcga_biolinks()

@install.command()
def install_meffil():
    """Installs meffil (update!)."""
    installer = PackageInstaller()
    installer.install_meffil()

@install.command()
def install_all_deps():
    """Installs bioconductor, minfi, enmix, tcga biolinks, and meffil."""
    installer = PackageInstaller()
    installer.install_bioconductor()
    installer.install_minfi_others()
    installer.install_tcga_biolinks()
    installer.install_meffil()



if __name__ == '__main__':
    install()
