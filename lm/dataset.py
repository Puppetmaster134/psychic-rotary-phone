import torch, re, collections
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

PAD_TOKEN = '&'
PAD_IDX = 0

SOS_TOKEN = '?'
SOS_IDX = 1

EOS_TOKEN = '@'
EOS_IDX = 2

SEP_TOKEN = '$'
SEP_IDX = 3

UNK_TOKEN = '%'
UNK_IDX = 4

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
            
            for i in tqdm(range(n_merges)):
                pairs = get_stats(split_words)
                best = max(pairs, key=pairs.get)
                split_words = merge_vocab(best, split_words)
            
            token_count = collections.defaultdict(int)
            for word in split_words:
                tokens = word.split()
                for token in tokens:
                    token_count[token] += 1
            
            self.token_count = token_count
            
            self.token_idx = {PAD_TOKEN: PAD_IDX, SOS_TOKEN : SOS_IDX, EOS_TOKEN : EOS_IDX, SEP_TOKEN: SEP_IDX, UNK_TOKEN : UNK_IDX}
            for idx, token in enumerate(self.token_count.keys(), len(self.token_idx)):
                self.token_idx[token] = idx
            
            self.idx_token = {idx : token for token, idx in self.token_idx.items()}
    
    def __len__(self):
        return len(self.token_idx)
    
    def encode(self, word):
        split_word = ' ' + ' '.join([*word]) + ' '
        for key in sorted(self.token_count.keys(), key=len, reverse=True):
            if len(key) > 1:
                split_token = ' '.join([*key])
                split_word = split_word.replace(f' {split_token} ', f' {key} ')

        tokens = [self.token_idx[token] for token in split_word.split()]
        return tokens

    def decode(self, indices, show_all=False):
        skip = set()
        if not show_all:
            skip = {PAD_IDX, SOS_IDX, EOS_IDX}
        
        return ''.join([self.idx_token[idx] for idx in indices if idx not in skip])


class DerivativeDataset(Dataset):
    def __init__(self, source_df : pd.DataFrame, tokenizer : BPETokenizer):
        self.data = source_df.to_numpy()
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x,y = self.data[idx]
        
        source = f'{SOS_TOKEN}{x}{SEP_TOKEN}{y}'
        target = f'{x}{SEP_TOKEN}{y}{EOS_TOKEN}'
        
        pred_ctx = f'{SOS_TOKEN}{x}{SEP_TOKEN}'
        
        source = self.tokenizer.encode(source)
        target = self.tokenizer.encode(target)
        pred_ctx = self.tokenizer.encode(pred_ctx)
        truth = y
        return {'source' : source, 'target': target, 'pred_ctx' : pred_ctx, 'truth': truth}


def batch_collator(batch):
    neg_attn = 0
    inputs = []
    targets = []
    contexts = []
    truths = []
    

    attention_mask = []
    for row in batch:
        input = torch.LongTensor(row['source'])
        target = torch.LongTensor(row['target'])
        mask = torch.ones_like(input)
        context = torch.LongTensor(row['pred_ctx'])
        
        inputs.append(input)
        targets.append(target)
        attention_mask.append(mask)
        contexts.append(context)
        truths.append(row['truth'])
    
    inputs = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX)
    targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=neg_attn)
    contexts = pad_sequence(contexts, batch_first=True, padding_value=PAD_IDX)
    

    return inputs, targets, attention_mask, contexts, truths











# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# def batch_collator(batch):
#     max_len = 32
#     neg_attn = 1
#     # neg_attn = float('-inf')
    
#     inputs = []
#     targets = []
#     truths = []
    
#     enc_mask = []
#     dec_mask = []
#     for row in batch:
#         enc_mask_single = [0 if i < len(row['x']) else neg_attn for i in range(max_len)]
#         enc_mask.append([enc_mask_single for _ in range(max_len)])
#         dec_mask.append([[0 if i <= idx else neg_attn for i in range(max_len)] if idx < len(row['y1']) else [neg_attn for _ in range(max_len)] for idx in range(max_len)])
#         x_pad = [row['x'][idx] if idx < len(row['x']) else PAD_IDX for idx in range(max_len)]
#         inputs.append(torch.LongTensor(x_pad))
        
#         y1_pad = [row['y1'][idx] if idx < len(row['y1']) else PAD_IDX for idx in range(max_len)]
#         targets.append(torch.LongTensor(y1_pad))
        
#         y2_pad = [row['y2'][idx] if idx < len(row['y2']) else PAD_IDX for idx in range(max_len)]
#         truths.append(torch.LongTensor(y2_pad))
    
#     enc_mask = torch.Tensor(enc_mask)
#     dec_mask = torch.Tensor(dec_mask)
#     inputs = torch.stack(inputs)
#     targets = torch.stack(targets)
#     truths = torch.stack(truths)
    
#     # inputs = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX)
#     # targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
#     # truths = pad_sequence(truths, batch_first=True, padding_value=PAD_IDX)
#     return inputs, targets, truths, enc_mask, dec_mask
    
# class DerivativeDataset(Dataset):
#     def __init__(self, source_df : pd.DataFrame, tokenizer : BPETokenizer):
#         self.data = source_df.to_numpy()
#         self.tokenizer = tokenizer
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         x,y = self.data[idx]
        
#         x_text = f'{SOS_TOKEN}{x}{EOS_TOKEN}'
#         y1_text = f'{SOS_TOKEN}{y}'
#         y2_text = f'{y}{EOS_TOKEN}'
#         x = self.tokenizer.encode(x_text)
#         y1 = self.tokenizer.encode(y1_text)
#         y2 = self.tokenizer.encode(y2_text)
        
#         return {'x':x, 'y1':y1, 'y2': y2}