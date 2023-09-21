import transformers as t
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class DerivativeDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        n_tokens = kwargs.get("n_tokens", 128)
        
        #set bos token id and eos token id and pad
        self.config = t.GPT2Config(
            vocab_size = n_tokens,
            n_positions = 64,
            n_embd = 24,
            # n_layer = 4,
            # n_head = 4
            # n_head = 2
        )
        
        self.gpt2 = t.GPT2LMHeadModel(self.config)
        
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, input, labels, attention_mask):
        out : CausalLMOutputWithCrossAttentions = self.gpt2(
            input_ids = input,
            attention_mask = attention_mask,
            labels = labels
        )
        
        return out
    
    def predict(self, input, sos_idx, max_tokens=32):
        current = input
        predictions = None
        with torch.no_grad():
            for i in range(max_tokens):
                mask = torch.where(current != 0, 1, 0)
                out : CausalLMOutputWithCrossAttentions = self.gpt2(
                    input_ids = current,
                    attention_mask = mask
                )
                
                logits = out.logits
                token_probs = self.softmax(logits)
                next_token = torch.argmax(token_probs, dim=-1)[:,-1:]

                current = torch.concat((current, next_token), dim=-1)
                predictions = next_token if predictions is None else torch.concat((predictions,next_token),dim=-1)
        return predictions