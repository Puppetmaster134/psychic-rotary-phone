import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import Dataset

PAD_IDX = 0
SOS_TOKEN = '?'
EOS_TOKEN = '@'
def encode(text : str):
    return [ord(c) for c in SOS_TOKEN + text + EOS_TOKEN]

def decode(indices : list):
    return ''.join([chr(i) for i in indices if i not in { ord(SOS_TOKEN), ord(EOS_TOKEN), PAD_IDX}])

def batch_collator(batch):
    inputs = []
    targets = []
    for x,y in batch:
        inputs.append(torch.LongTensor(x))
        targets.append(torch.LongTensor(y))
    
    inputs = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX)
    targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    return inputs, targets
    
class DerivativeDataset(Dataset):
    def __init__(self, source_df : pd.DataFrame):
        self.data = source_df.to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x,y = self.data[idx]
        return encode(x), encode(y)