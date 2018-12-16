import torch
from torch import cat, stack, FloatTensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Scale, Pad
import pandas as pd, numpy as np

def _reshape(cpgs_per_row, l):
    def resize(vector):
        return vector[:,:int(l[1]/cpgs_per_row)*cpgs_per_row].reshape((l[0],int(l[1]/cpgs_per_row),cpgs_per_row))
    return resize

def convert_to_tensor():
    def to_tensor(arr):
        out= torch.FloatTensor(arr)
        return out
    return to_tensor

class Transformer:
    def __init__(self, convolutional=False, cpg_per_row=30000, l=None):
        self.convolutional = convolutional
        self.cpg_per_row = cpg_per_row
        self.l = l
        if convolutional:
            self.shape=(self.l[0],int(self.l[1]/cpgs_per_row),cpgs_per_row)
        else:
            self.shape=None

    def generate(self):
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

def get_methylation_dataset(methylation_array, outcome_col, convolutional=False, cpg_per_row=1200):
    return MethylationDataSet(methylation_array, Transformer(convolutional, cpg_per_row, methylation_array.beta.shape), outcome_col)

class MethylationDataSet(Dataset):
    def __init__(self, methylation_array, transform, outcome_col='', mlp=False):
        self.methylation_array = methylation_array
        self.outcome_col = self.methylation_array.pheno[outcome_col] if outcome_col else pd.Series(np.ones(len(self)),index=self.methylation_array.pheno.index)
        self.outcome_col = self.outcome_col.loc[self.methylation_array.beta.index,]
        self.samples = np.array(list(self.outcome_col.index))
        self.transform = transform
        self.new_shape = self.transform.shape
        self.mlp=mlp

    def __getitem__(self,index):
        if self.mlp:
            return self.get_item_mlp(index)
        else:
            return self.get_item_vae(index)

    def get_item_vae(self, index):# .iloc[index,] [index] .iloc[index]
        transform = self.transform.generate()
        #print(self.methylation_array.beta.values.shape)
        return transform(self.methylation_array.beta.values),self.samples.tolist(),self.outcome_col.values.tolist()

    def get_item_mlp(self, index):# .iloc[index,] [index] .iloc[index]
        transform = self.transform.generate()
        #print(self.methylation_array.beta.values.shape)
        return transform(self.methylation_array.beta.iloc[index,:].values),self.samples[index],self.outcome_col.iloc[index,:].values

    def __len__(self):
        return 1 if not self.mlp else self.methylation_array.beta.shape[0]
