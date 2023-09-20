import torch, re, collections
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import Dataset

PAD_TOKEN = '&'
PAD_IDX = 0

SOS_TOKEN = '?'
SOS_IDX = 1

EOS_TOKEN = '@'
EOS_IDX = 2

UNK_TOKEN = '%'
UNK_IDX = 3

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word in vocab:
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += 1
    return pairs

def merge_vocab(pair, v_in):
    v_out = []
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        v_out.append(p.sub(''.join(pair), word))
    return v_out

class BPETokenizer:
    def __init__(self, corpus = None, n_merges=15):
        
        if corpus:
            split_words = [' '.join([*element]) for element in corpus]
            
            for i in range(n_merges):
                pairs = get_stats(split_words)
                best = max(pairs, key=pairs.get)
                split_words = merge_vocab(best, split_words)
                print(best)
            
            token_count = collections.defaultdict(int)
            for word in split_words:
                tokens = word.split()
                for token in tokens:
                    token_count[token] += 1
            
            self.token_count = token_count
            
            self.token_idx = {PAD_TOKEN: PAD_IDX, SOS_TOKEN : SOS_IDX, EOS_TOKEN : EOS_IDX, UNK_TOKEN : UNK_IDX}
            for idx, token in enumerate(self.token_count.keys(), len(self.token_idx)):
                self.token_idx[token] = idx
            
            self.idx_token = {idx : token for token, idx in self.token_idx.items()}
    
    def __len__(self):
        return len(self.token_idx)
    
    def encode(self, word):
        formatted_word = f'{SOS_TOKEN}{word}{EOS_TOKEN}'
        split_word = ' '.join([*formatted_word])
        for key in sorted(self.token_count.keys(), key=len, reverse=True):
            if len(key) > 1:
                split_token = ' '.join([*key])
                split_word = split_word.replace(f' {split_token} ', f' {key} ')

        tokens = [self.token_idx[token] for token in split_word.split()]
        
        # while len(tokens) < 32:
        #     tokens.append(PAD_IDX)
            
        return tokens

    def decode(self, indices):
        return ''.join([self.idx_token[idx] for idx in indices if idx not in {PAD_IDX, SOS_IDX, EOS_IDX}])


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
    def __init__(self, source_df : pd.DataFrame, tokenizer : BPETokenizer):
        self.data = source_df.to_numpy()
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x,y = self.data[idx]
        
        return self.tokenizer.encode(x), self.tokenizer.encode(y)