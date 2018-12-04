import torch
from torch import cat, stack, FloatTensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Scale, Pad
import pandas as pd, numpy as np

class Transformer:
    def __init__(self):
        pass

    def generate(self):
        return Compose([ToTensor()])

# From Titus
# test_set_percent = 0.1
# methyl_test_df = methyl_df2.sample(frac=test_set_percent)
# methyl_train_df = methyl_df2.drop(methyl_test_df.index)
# instead get beta keys and sample

def get_methylation_dataset(methylation_array, outcome_col):
    return MethylationDataSet(methylation_array, Transformer(), outcome_col)

class MethylationDataSet(Dataset):
    def __init__(self, methylation_array, transform, outcome_col=''):
        self.methylation_array = methylation_array
        self.outcome_col = self.methylation_array.pheno[outcome_col] if outcome_col else pd.Series(np.ones(len(self)),index=self.methylation_array.pheno.index)
        self.outcome_col = self.outcome_col.loc[self.methylation_array.beta.index,]
        self.samples = np.array(list(self.outcome_col.index))
        self.transform = transform

    def __getitem__(self, index):
        transform = self.transform.generate()
        return transform(self.methylation_array.beta.iloc[index,].values),self.samples[index],self.outcome_col.iloc[index]

    def __len__(self):
        return self.methylation_array.beta.shape[0]
