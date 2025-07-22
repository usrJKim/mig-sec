import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncodingk, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #Positinal encodings
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torchy.exp(torch.arange(0, model_dim, 2) *
                              -(math.log(10000.0)/model_dim))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        # TODO

class simpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout = 0.1):
        super(simpleTransformer, self).__init__() #initialize torch nn.Module
        self.embedding = nn.Embedding(input_dim, model_dim) # embedding


        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) #(B, T, D)
        x = self.transformer_encoder(x) #(B,T,D)
        x = x.mean(dim=1) #mean pooling
        return self.classifier(x)

    def generate(self, input_ids, max_len=20):
        self.eval()
        generated = input_ids
        for _ in range(max_len):
            with torch.no_grad():
                out = self.forward(generated)
                next_token = out.argmax(dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)
        return generated

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_k=None, past_v=None, mask=None):
        B, T, D = x.shape

        # Project inputs to Q, K, V
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: (B, num_heads, T, head_dim)
        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # KV cache: concat with past
        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=2)  # (B, num_heads, T_total, head_dim)
            v = torch.cat([past_v, v], dim=2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, T, T_kv)
        attn_scores /= self.head_dim ** 0.5

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        out = self.out_proj(out)

        return out, k, v

class GPTBlock(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(GPTBlock, self).__init__()
        self.attn = CustomMultiHeadAttention(model_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, 4*model_dim),
            nn.ReLU(),
            nn.Linear(4* model_dim, model_dim),
        )
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, past_kv=None):
        #x: (B,T,D), past_kv: tuple(k,v)
        past_k, past_v = past_kv if past_kv is not None else (None, None)
        B, T, D = x.shape

        # causal mask
        device = x.device
        mask = torch.tril(torch.ones(T,T, device=device)).unsqueeze(0).unsqueeze(0) #(1,1,T,T)

        attn_out, k, v = self.attn(x, past_k, past_v, mask)
        x = self.ln1(x + self.dropout(attn_out))
        x = self.ln2(x + self.dropout(self.ff(x)))
        return x, (k, v) 

class SimpleGPT(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, max_len=512, num_classes=2):
        super(SimpleGPT,self).__init__()
        self.token_emb = nn.Embedding(input_dim, model_dim)
        self.pos_emb = nn.Embedding(max_len, model_dim)

        self.blocks = nn.ModuleList([
                                        GPTBlock(model_dim, num_heads) for _ in range(num_layers)
                                    ])

        self.ln_f = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B,T)
        x = self.token_emb(x) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x, None)

        x = self.ln_f(x)
        x = x[:, -1, :] #use last output token
        return self.classifier(x)

    def generate(self, input_ids, max_len = 20, use_kv_cache = True):
        self.eval()
        generated = input_ids # shape: (B, T0)
        B, T= input_ids.shape
        device = input_ids.device

        if use_kv_cache:
            # 1. prefill
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            x = self.token_emb(input_ids) + self.pos_emb(positions)

            past_kvs = []
            for i, block in enumerate(self.blocks):
                x, kv = block(x, past_kv=None)
                past_kvs.append(kv)
            x = self.ln_f(x)
            logits = self.classifier(x[:,-1,:])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = next_token

            # 2. decode loop
            for step in range(max_len-1):
                positions = torch.tensor([[T + step]], device=device).expand(B,1)

                x = self.token_emb(next_token) + self.pos_emb(positions)

                new_kvs = []
                for i, block in enumerate(self.blocks):
                    past_k, past_v = past_kvs[i]
                    x, (k_new, v_new) = block(x, (past_k, past_v))
                    new_k = torch.cat([past_k, k_new], dim=2) if past_k is not None else k_new
                    new_v = torch.cat([past_v, v_new], dim=2) if past_v is not None else v_new
                    new_kvs.append((new_k, new_v))
                past_kvs = new_kvs

                x = self.ln_f(x)
                logits = self.classifier(x[:, -1, :])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        else:
            generated = input_ids.clone()
            for step in range(max_len):
                T_all = generated.size(1)
                positions = torch.arange(T_all, device=device).unsqueeze(0).expand(B, T_all)
                x = self.token_emb(generated) + self.pos_emb(positions)

                for step, block in enumerate(self.blocks):
                    x, _ = block(x, past_kv=None)

                x = self.ln_f(x)
                logits = self.classifier(x[:,-1,:])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim = 1)
            generated = generated[:, input_ids.size(1):]
        return generated

def get_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train")
    parser.add_argument("--model", type=str, choices=["transformer", "gpt"])
    parser.add_argument("--usekv", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=1000)

    return parser.parse_args()

def train(args, model):
    print("Training...")
    dummy_input = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len)).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(args.epochs):
        outputs = model(dummy_input)
        targets = torch.randint(0, 2, (args.batch_size,)).to("cuda")
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch: {epoch+1}/{args.epochs}")

def infer(args, model):
    print("Inference...")
    dummy_input = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len)).to("cuda")
    model.eval()
    with torch.no_grad():
        if args.model == "gpt":
            if args.usekv == "True":
                model.generate(dummy_input,max_len=20, True)
            else:
                model.generate(dummy_input,max_len=20, False)
        else:
            model.generate(dummy_input,max_len=20)
        print("Done!")

if __name__ == '__main__':
    args = get_args()

    if args.model =="transformer":
        model = simpleTransformer(
            input_dim = args.vocab_size,
            model_dim = args.model_dim,
            num_heads = args.num_heads,
            num_layers=args.num_layers,
            num_classes=2,
        ).to("cuda")
        
    elif args.model == "gpt":
        model = SimpleGPT(
            input_dim = args.vocab_size,
            model_dim = args.model_dim,
            num_heads = args.num_heads,
            num_layers = args.num_layers,
            num_classes = 2
        ).to("cuda")

    # TODO Implement KV cache, longer input text
    if args.mode == "train":
        train(args, model)
    elif args.mode == "infer":
        infer(args, model)
    
