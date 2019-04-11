"""
datasets.py
=======================
Contains datatypes core to loading MethylationArrays into Torch tensors.
"""

import torch
from torch import cat, stack, FloatTensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Scale, Pad
import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder
from pymethylprocess.MethylationDataTypes import MethylationArray

def _reshape(cpgs_per_row, l):
    """Reshapes one sample's methylation values into image.

    Parameters
    ----------
    cpgs_per_row : int
        Number of CpGs per row
    l : int
        Number of splits.
    """
    def resize(vector):
        return vector[:,:int(l[1]/cpgs_per_row)*cpgs_per_row].reshape((l[0],int(l[1]/cpgs_per_row),cpgs_per_row))
    return resize

def convert_to_tensor():
    def to_tensor(arr):
        out= torch.FloatTensor(arr)
        return out
    return to_tensor

class Transformer:
    """Short summary.

    Parameters
    ----------
    convolutional : type
        Description of parameter `convolutional`.
    cpg_per_row : type
        Description of parameter `cpg_per_row`.
    l : type
        Description of parameter `l`.

    Attributes
    ----------
    shape : type
        Description of attribute `shape`.
    convolutional
    cpg_per_row
    l

    """
    def __init__(self, convolutional=False, cpg_per_row=30000, l=None):
        self.convolutional = convolutional
        self.cpg_per_row = cpg_per_row
        self.l = l
        if convolutional:
            self.shape=(self.l[0],int(self.l[1]/cpgs_per_row),cpgs_per_row)
        else:
            self.shape=None

    def generate(self):
        """Short summary.

        Parameters
        ----------


        Returns
        -------
        type
            Description of returned object.

        """
        data_aug = []
        if self.convolutional and self.l != None:
            data_aug.append(_reshape(self.cpgs_per_row, self.l))
        data_aug.append(convert_to_tensor())
        return Compose(data_aug)

# From Titus
# test_set_percent = 0.1
# methyl_test_df = methyl_df2.sample(frac=test_set_percent)
# methyl_train_df = methyl_df2.drop(methyl_test_df.index)
# instead get beta keys and sample


def get_methylation_dataset(methylation_array, outcome_col, convolutional=False, cpg_per_row=1200, predict=False, categorical=False, categorical_encoder=False):
    """Short summary.

    Parameters
    ----------
    methylation_array : type
        Description of parameter `methylation_array`.
    outcome_col : type
        Description of parameter `outcome_col`.
    convolutional : type
        Description of parameter `convolutional`.
    cpg_per_row : type
        Description of parameter `cpg_per_row`.
    predict : type
        Description of parameter `predict`.
    categorical : type
        Description of parameter `categorical`.
    categorical_encoder : type
        Description of parameter `categorical_encoder`.

    Returns
    -------
    type
        Description of returned object.

    """
    if predict:
        return MethylationPredictionDataSet(methylation_array, Transformer(convolutional, cpg_per_row, methylation_array.beta.shape), outcome_col, categorical=categorical, categorical_encoder=categorical_encoder)
    else:
        return MethylationDataSet(methylation_array, Transformer(convolutional, cpg_per_row, methylation_array.beta.shape), outcome_col)

class MethylationDataSet(Dataset):
    """Short summary.

    Parameters
    ----------
    methylation_array : type
        Description of parameter `methylation_array`.
    transform : type
        Description of parameter `transform`.
    outcome_col : type
        Description of parameter `outcome_col`.
    categorical : type
        Description of parameter `categorical`.
    categorical_encoder : type
        Description of parameter `categorical_encoder`.

    Attributes
    ----------
    samples : type
        Description of attribute `samples`.
    features : type
        Description of attribute `features`.
    encoder : type
        Description of attribute `encoder`.
    new_shape : type
        Description of attribute `new_shape`.
    length : type
        Description of attribute `length`.
    methylation_array
    outcome_col
    transform

    """
    def __init__(self, methylation_array, transform, outcome_col='', categorical=False, categorical_encoder=False):
        self.methylation_array = methylation_array
        self.outcome_col = self.methylation_array.pheno.loc[:,outcome_col] if outcome_col else pd.Series(np.ones(len(self)),index=self.methylation_array.pheno.index)
        self.outcome_col = self.outcome_col.loc[self.methylation_array.beta.index,]
        self.samples = np.array(list(self.methylation_array.beta.index))
        self.features = np.array(list(self.methylation_array.beta))
        self.methylation_array.beta = self.methylation_array.beta.values
        self.samples = np.array(list(self.outcome_col.index))
        self.outcome_col=self.outcome_col.values
        if categorical:
            #self.outcome_col=self.outcome_col#[:,np.newaxis]
            if not categorical_encoder:
                print(self.outcome_col)
                self.encoder = OneHotEncoder(sparse=False)
                self.encoder.fit(self.outcome_col)
            else:
                self.encoder=categorical_encoder
            self.outcome_col=self.encoder.transform(self.outcome_col)
        else:
            self.encoder = None
        self.transform = transform
        self.new_shape = self.transform.shape
        self.length = self.methylation_array.beta.shape[0]
        print(self.outcome_col)
        print(self.outcome_col.shape)

    def to_methyl_array(self):
        """Short summary.

        Parameters
        ----------


        Returns
        -------
        type
            Description of returned object.

        """
        return MethylationArray(self.methylation_array.pheno,pd.DataFrame(self.methylation_array.beta,index=self.samples,columns=self.features),'')

    def __getitem__(self,index):
        return self.transform.generate()(self.methylation_array.beta[index,:]),self.samples[index],self.outcome_col[index]
        """if self.mlp:
            return self.get_item_mlp(index)
        else:
            return self.get_item_vae(index)"""

    """def get_item_vae(self, index):# .iloc[index,] [index] .iloc[index]
        transform = self.transform.generate()
        #print(self.methylation_array.beta.values.shape)
        return transform(self.methylation_array.beta.values),self.samples.tolist(),self.outcome_col.values.tolist()

    def get_item_mlp(self, index):# .iloc[index,] [index] .iloc[index]
        transform = self.transform.generate()
        #print(self.methylation_array.beta.values.shape)
        return transform(self.methylation_array.beta.iloc[index,:].values),self.samples[index],self.outcome_col.iloc[index,:].values
        """
    def __len__(self):
        return self.length

class MethylationPredictionDataSet(MethylationDataSet):
    """Short summary.

    Parameters
    ----------
    methylation_array : type
        Description of parameter `methylation_array`.
    transform : type
        Description of parameter `transform`.
    outcome_col : type
        Description of parameter `outcome_col`.
    categorical : type
        Description of parameter `categorical`.
    categorical_encoder : type
        Description of parameter `categorical_encoder`.

    Attributes
    ----------
    self,index : type
        Description of attribute `self,index`.
    outcome_col

    """
    def __init__(self, methylation_array, transform, outcome_col='', categorical=False, categorical_encoder=False):
        super().__init__(methylation_array, transform, outcome_col, categorical, categorical_encoder=categorical_encoder)
        print(self.outcome_col)

    def __getitem__(self,index):
        return self.transform.generate()(self.methylation_array.beta[index,:]),self.samples[index],torch.FloatTensor(self.outcome_col[index,:])

class RawBetaArrayDataSet(Dataset):
    """Short summary.

    Parameters
    ----------
    beta_array : type
        Description of parameter `beta_array`.
    transform : type
        Description of parameter `transform`.

    Attributes
    ----------
    length : type
        Description of attribute `length`.
    self,index : type
        Description of attribute `self,index`.
    beta_array
    transform

    """
    def __init__(self, beta_array, transform):
        self.beta_array = beta_array
        self.transform = transform
        self.length = self.beta_array.shape[0]

    def __getitem__(self,index):
        return self.transform.generate()(self.beta_array[index,:])

    def __len__(self):
        return self.length
