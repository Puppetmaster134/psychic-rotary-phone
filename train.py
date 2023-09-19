import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import NLLLoss

from lm.model import DerivativeSolver
from lm.dataset import DerivativeDataset, batch_collator, decode, PAD_IDX
import numpy as np

from tqdm import tqdm

DATA_DIR = './data/'
DEVICE = 'cuda'
LEARNING_RATE = 5e-5

def load_datasets():
    train_set = pd.read_json(DATA_DIR + "train.json")
    test_set = pd.read_json(DATA_DIR + "test.json")
    return DerivativeDataset(train_set), DerivativeDataset(test_set)

def train(model, dataset):
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    obj = NLLLoss(ignore_index=PAD_IDX)
    dataloader = DataLoader(dataset, batch_size = 16, collate_fn=batch_collator)
    
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
            tqdm.write(decode(pred))
            tqdm.write(decode(tgt))
            tqdm.write(f"Batch {idx} -- {total_loss / (idx + 1)}")

    epoch_avg_loss = total_loss / len(dataloader)
    return epoch_avg_loss

def test(model, dataset):
    dataloader = DataLoader(dataset, batch_size = 32, collate_fn=batch_collator)
    
    scores = []
    with torch.no_grad():
        for idx, (source, target) in enumerate(tqdm(dataloader,total=len(dataloader))):
            source = source.to(DEVICE)
            target = target.to(DEVICE)
            outputs,_,_ = model(source)
            
            print(target.size())
            t = outputs.size()
            _,indexes = torch.topk(outputs,1,dim=-1)
            results = indexes.squeeze(-1)
            
            
            r = results.cpu().numpy()
            t = target.cpu().numpy()
            for rp, tp in zip(r,t):
                first = decode(rp)
                second = decode(tp)
                print(first)
                print(second)
                print()
                scores.append(int(first == second))
    
    return np.mean(scores)   
train_set, test_set = load_datasets()
model = DerivativeSolver().to(DEVICE)
for i in range(20):
    loss = train(model, train_set)
    print(f'Epoch {i} loss', loss)
    
accuracy = test(model, test_set)
print(f'Epoch {i} accuracy', accuracy)
