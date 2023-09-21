import pickle, os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import NLLLoss

from lm.model import DerivativeSolver, TestCity
from lm.dataset import DerivativeDataset, batch_collator, PAD_IDX, SOS_IDX
import numpy as np

from tqdm import tqdm
from lm.dataset import BPETokenizer

from lm.numodel import DerivativeDecoder

DATA_DIR = './data/'
MODEL_DIR = 'saved_models/'
DEVICE = 'cuda'
LEARNING_RATE = 5e-5
LEARNING_DECAY = 0.95
BATCH_SIZE = 32
NUM_EPOCHS = 5

def load_tokenizer(token_elements):
    filename = MODEL_DIR + "tokenizer.pt"
    
    if os.path.isfile(filename):
        with open(filename,'rb') as f:
            tokenizer : BPETokenizer = pickle.load(f)
    else:
        tokenizer = BPETokenizer(token_elements,n_merges=15)
        with open(filename, 'wb') as f:
            pickle.dump(tokenizer, f)
    
    return tokenizer
    
def load_datasets():
    train_set = pd.read_json(DATA_DIR + "train.json")
    test_set = pd.read_json(DATA_DIR + "test.json")
    token_elements = train_set['x'].tolist() + train_set['y'].tolist()
    tokenizer = load_tokenizer(token_elements)
    
    return DerivativeDataset(train_set, tokenizer), DerivativeDataset(test_set, tokenizer)
# def train(model : DerivativeSolver, dataset : DerivativeDataset, optimizer):
#     obj = NLLLoss(ignore_index=PAD_IDX)
#     dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, collate_fn=batch_collator)
    
#     total_loss = 0
#     for idx, (source, target, truth, enc_mask, dec_mask) in enumerate(tqdm(dataloader,total=len(dataloader))):
#         optimizer.zero_grad()
#         source = source.to(DEVICE)
#         target = target.to(DEVICE)
#         truth = truth.to(DEVICE)
#         enc_mask = enc_mask.to(DEVICE)
#         dec_mask = dec_mask.to(DEVICE)

#         # outputs,_,_ = model(source, target, enc_mask, dec_mask)
#         outputs,_,_ = model(source, target)
        
#         loss = obj(outputs.view(-1, outputs.size(-1)),truth.view(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
        
#         if idx % 250 == 249:
#             tqdm.write(f'Loss {idx} -- {total_loss / idx}')
#     epoch_avg_loss = total_loss / len(dataloader)
#     return epoch_avg_loss

def train(model : DerivativeDecoder, dataset : DerivativeDataset, optimizer):
    obj = NLLLoss(ignore_index = PAD_IDX)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=batch_collator)
    total_loss = 0
    for idx, (input, target, mask,_,_) in enumerate(tqdm(dataloader,total=len(dataloader))):
        input = input.to(DEVICE)
        target = target.to(DEVICE)
        mask = target.to(DEVICE)

        out = model(input = input, labels = target, attention_mask = mask)
        
        loss = out.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if idx % 250 == 249:
            tqdm.write(f'Loss {idx} -- {total_loss / idx}')
            
    epoch_avg_loss = total_loss / len(dataloader)
    return epoch_avg_loss
        
    

def test(model, dataset : DerivativeDataset):
    scores = []
    for row in tqdm(dataset, total=len(dataset)):
        input = torch.LongTensor(row['pred_ctx']).unsqueeze(0).to(DEVICE)
        preds = model.predict(input,SOS_IDX).flatten().cpu().numpy()
        truth = row['truth']
        prediction = dataset.tokenizer.decode(preds)
        score = 1 if truth == prediction else 0
        scores.append(score)
        
    tqdm.write(f'{truth} || {prediction}')
    return np.mean(scores)

def save_model(model):
    torch.save(model.state_dict(), MODEL_DIR + 'best.pt')
    
train_set, test_set = load_datasets()
n_tokens = len(train_set.tokenizer.idx_token)

# model = DerivativeSolver(
#     n_tokens=n_tokens,
#     hidden_dim = 256,
#     dropout = 0.25,
#     tf_ratio = 0.25
# ).to(DEVICE)

from lm.numodel import DerivativeDecoder
model = DerivativeDecoder(
    n_tokens = n_tokens
).to(DEVICE)


# model = TestCity(n_tokens=n_tokens).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameter Count: {total_params}")


NUM_EPOCHS = 20
best = 0
for i in range(NUM_EPOCHS):
    # Train
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=LEARNING_DECAY)
    loss = train(model, train_set, optimizer)
    print(f'Epoch {i} loss', loss)
    scheduler.step()
    
    # Test
    accuracy = test(model, test_set)
    print(f'Epoch {i} accuracy', accuracy)
    
    if accuracy >= best:
        best = accuracy
        print('New best accuracy. Saving.')
        save_model(model)
