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

from lm.tmp import Transformer

DATA_DIR = './data/'
MODEL_DIR = 'saved_models/'
DEVICE = 'cuda'
LEARNING_RATE = 5e-3
LEARNING_DECAY = 0.99
BATCH_SIZE = 128
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
    train_set = pd.read_json(DATA_DIR + "train.json").head(5000)
    test_set = pd.read_json(DATA_DIR + "test.json")
    token_elements = train_set['x'].tolist() + train_set['y'].tolist()
    tokenizer = load_tokenizer(token_elements)
    
    return DerivativeDataset(train_set, tokenizer), DerivativeDataset(test_set, tokenizer)









#GPT2 Trainer
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
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if idx % 250 == 249:
            tqdm.write(f'Loss {idx} -- {total_loss / idx}')
            
    epoch_avg_loss = total_loss / len(dataloader)
    return epoch_avg_loss   
#Tester
def test(model, dataset : DerivativeDataset):
    scores = []
    sample_idx = np.random.default_rng().choice(len(dataset),size=200, replace=False)
    for idx in tqdm(sample_idx):
        row = dataset[idx]
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



def train_loop(model,dataset, opt):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=batch_collator)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        X, y, _,_,_ = batch
        
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        
        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:,:-1]
        y_expected = y[:,1:]
        
        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(DEVICE)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)      
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)


train_set, test_set = load_datasets()
n_tokens = len(train_set.tokenizer.idx_token)




# model = Transformer(
#     num_tokens=n_tokens, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
# ).to(DEVICE)
# opt = torch.optim.SGD(model.parameters(), lr=0.01)

model = DerivativeDecoder(
    n_tokens = n_tokens
).to(DEVICE)

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
print(f'Accuracy', accuracy)

if accuracy >= best:
    best = accuracy
    print('New best accuracy. Saving.')
    save_model(model)
