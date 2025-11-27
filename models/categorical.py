from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor, Tensor, nn
from models.utils import mean_by_quarter, mean_by_mod4
from models.layer_norm import LayerNorm

@dataclass
class CategoricalConfig:
    n_block: int = 1024
    n_vocab: int = 50304
    n_embd: int = 768
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    dropout: float = 0.0

class Categorical(nn.Module):
    def __init__(self, config, backbone: nn.Module):
        super().__init__()
        assert config.n_vocab is not None
        assert config.n_block is not None
        self.n_block = config.n_block

        self.wte = nn.Embedding(config.n_vocab, config.n_embd)
        self.wpe = nn.Embedding(config.n_block, config.n_embd)
        self.backbone = backbone
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, config.n_vocab, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, state, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.n_block, f"Cannot forward sequence of length {t}, block size is only {self.n_block}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        x, state = self.backbone(x, state)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)                        
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, state, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, state, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at n_block
            idx_cond = idx if idx.size(1) <= self.n_block else idx[:, -self.n_block:]
            # forward the model to get the logits for the index in the sequence
            logits, state, _ = self(idx_cond, state)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def initial_state(self, batch_size):
        return self.backbone.initial_state(batch_size)

    def flops_per_token(self):
        return self.backbone.flops_per_token()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params
