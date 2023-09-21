import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lm.dataset import SOS_IDX

DEVICE = 'cuda'
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, tf_ratio = 0.0):
        super(Decoder, self).__init__()
        self.tf_rate = tf_ratio
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, max_length = 30):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_IDX).to(DEVICE)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None and np.random.rand() < self.tf_rate :
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

class DerivativeSolver(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        n_tokens = kwargs.get("n_tokens", 128)
        dropout = kwargs.get("dropout", 0.0)
        hidden_dim = kwargs.get("hidden_dim", 512)
        tf_ratio = kwargs.get("tf_ratio", 0.0)
        
        self.encoder = Encoder(
            input_size = n_tokens,
            hidden_size = hidden_dim,
            dropout = dropout
        )
        
        self.decoder = Decoder(
            output_size = n_tokens,
            hidden_size = hidden_dim, 
            dropout = dropout,
            tf_ratio = tf_ratio
        )
    
    def forward(self, x, y=None):
        encoder_outputs, encoder_hidden = self.encoder(x)
        
        max_size = y.size(-1) if y is not None else 32
        decoder_outputs, decoder_hidden, attn = self.decoder(encoder_outputs, encoder_hidden, y, max_length=max_size)
        return decoder_outputs, decoder_hidden, attn

class TestCity(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        n_tokens = kwargs.get("n_tokens", 128)
        dropout = kwargs.get("dropout", 0.0)
        hidden_dim = kwargs.get("hidden_dim", 512)
        tf_ratio = kwargs.get("tf_ratio", 0.0)
        
        emb_dim = 16
        ff_dim = 64
        self.encoder_embed = nn.Embedding(n_tokens, emb_dim)
        self.decoder_embed = nn.Embedding(n_tokens, emb_dim)
        self.t = nn.Transformer(
            d_model = emb_dim,
            nhead = 1,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        
        self.lm_head = nn.Sequential(
            nn.Linear(emb_dim,n_tokens),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x, y, enc_mask, dec_mask):
        x_emb = self.encoder_embed(x)
        y_emb = self.decoder_embed(y)
        
        # print(dec_mask)
        out = self.t(x_emb, y_emb, src_mask=enc_mask, tgt_mask = dec_mask)
        
        preds = self.lm_head(out)
        return preds,None,None
    
    def predict(self, x, enc_mask, n_tokens=32):
        with torch.no_grad():
            batch_size = x.size(0)
            y = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_IDX).to(DEVICE)
            x_emb = self.encoder_embed(x)
            
            for i in range(n_tokens-1):
                y_emb = self.decoder_embed(y)
                dec_mask = torch.zeros((batch_size, y.size(-1), y.size(-1))).to(DEVICE)
                out = self.t(x_emb, y_emb, src_mask=enc_mask, tgt_mask=dec_mask)
                pred = self.lm_head(out)
                idx = torch.argmax(pred, dim=-1)
                sos_tokens = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_IDX).to(DEVICE)
                y = torch.concat([sos_tokens,idx], dim=-1)

            return y

        
        # for i in range(31):
        #     self.transformer(y)
        
        # # # while decoder_input.
        # # # print(decoder_input.size())
        # # exit()
        # # pass