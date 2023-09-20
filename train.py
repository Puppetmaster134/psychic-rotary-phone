import pickle, os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import NLLLoss

from lm.model import DerivativeSolver
from lm.dataset import DerivativeDataset, batch_collator, PAD_IDX
import numpy as np

from tqdm import tqdm
from lm.dataset import BPETokenizer

DATA_DIR = './data/'
MODEL_DIR = 'saved_models/'
DEVICE = 'cuda'
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

def load_tokenizer(token_elements):
    filename = MODEL_DIR + "tokenizer.pt"
    
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            tokenizer : BPETokenizer = pickle.load(f)
    else:
        tokenizer = BPETokenizer(token_elements,n_merges=1)
        with open(filename, 'wb') as f:
            pickle.dump(tokenizer, f)
    
    return tokenizer
    
def load_datasets():
    train_set = pd.read_json(DATA_DIR + "train.json")
    test_set = pd.read_json(DATA_DIR + "test.json")    
    token_elements = train_set['x'].tolist() + train_set['y'].tolist()
    tokenizer = load_tokenizer(token_elements)
    
    return DerivativeDataset(train_set, tokenizer), DerivativeDataset(test_set, tokenizer)

def train(model : DerivativeSolver, dataset : DerivativeDataset, optimizer):
    obj = NLLLoss(ignore_index=PAD_IDX)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, collate_fn=batch_collator)
    
    total_loss = 0
    for idx, (source, target) in enumerate(tqdm(dataloader,total=len(dataloader))):
        optimizer.zero_grad()
        source = source.to(DEVICE)
        target = target.to(DEVICE)
        
        outputs,_,_ = model(source, target)

        loss = obj(outputs.view(-1, outputs.size(-1)),target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if idx % 500 == 0:
            _,indexes = torch.topk(outputs,1)
            pred = indexes.squeeze(-1).cpu().numpy()[0]
            tgt = target.squeeze(-1).cpu().numpy()[0]
            tqdm.write(dataset.tokenizer.decode(pred))
            tqdm.write(dataset.tokenizer.decode(tgt))
            tqdm.write(f"Batch {idx} -- {total_loss / (idx + 1)}")

    epoch_avg_loss = total_loss / len(dataloader)
    return epoch_avg_loss

def test(model, dataset):
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, collate_fn=batch_collator)
    
    scores = []
    with torch.no_grad():
        for idx, (source, target) in enumerate(tqdm(dataloader,total=len(dataloader))):
            source = source.to(DEVICE)
            target = target.to(DEVICE)
            outputs,_,_ = model(source)
            
            _,indexes = torch.topk(outputs,1,dim=-1)
            results = indexes.squeeze(-1)
            
            
            r = results.cpu().numpy()
            t = target.cpu().numpy()
            
            for rp, tp in zip(r,t):
                first = dataset.tokenizer.decode(rp)
                second = dataset.tokenizer.decode(tp)
                print(first)
                print(second)
                print()
                scores.append(int(first == second))
    
    return np.mean(scores)   

def save_model(model):
    torch.save(model.state_dict(), MODEL_DIR + 'best.pt')
    
train_set, test_set = load_datasets()
n_tokens = len(train_set.tokenizer.idx_token)

model = DerivativeSolver(
    n_tokens=n_tokens,
    hidden_dim = 256,
    dropout = 0.25,
    tf_ratio = 0.25
).to(DEVICE)

NUM_EPOCHS = 5
best = 0
for i in range(NUM_EPOCHS):
    # Train
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    loss = train(model, train_set, optimizer)
    print(f'Epoch {i} loss', loss)
    scheduler.step()
    
    # Test
    accuracy = test(model, test_set)
    print(f'Epoch {i} accuracy', accuracy)
    
    if accuracy > best:
        best = accuracy
        print('New best accuracy. Saving.')
        save_model(model)
