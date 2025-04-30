import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

def create_dictionary(smiles: list):
    d = {'PAD': 0, 'SOS': 1, 'EOS': 2}
    idx = 3
    chars = sorted(set(char for smile in smiles for char in smile))
    for char in chars:
        d[char] = idx
        idx += 1
    return d

def get_max_seqlen(smiles: list):
    return max(len(smile) for smile in smiles)

class SmilesDataset(Dataset):
    def __init__(self, path_to_file):
        self.smiles = []
        with open(path_to_file, "r") as f:
            self.smiles = [line.strip() for line in f.readlines()]
        
        self.dictionary = create_dictionary(self.smiles)
        self.max_seq_len = get_max_seqlen(self.smiles) + 2  # +SOS, EOS
        self.vocab_size = len(self.dictionary)
    
    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        x = np.zeros(self.max_seq_len, dtype=np.int64)
        x[0] = self.dictionary['SOS']
        smile = self.smiles[index]
        for i, c in enumerate(smile, start=1):
            x[i] = self.dictionary[c]
        x[len(smile)] = self.dictionary['EOS']
        padding_mask = (x == 0).astype(np.int64)
        return torch.tensor(x), torch.tensor(padding_mask).float()
        
        
class SmilesTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, dim=256, nhead=8, num_layers=6, dim_feedforward=1024, max_seq_length=101):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_encoding = nn.Parameter(self.generate_positional_encoding(max_seq_length, dim), requires_grad=False)

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(dim, vocab_size)

    def generate_positional_encoding(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, padding_mask, look_ahead_mask, device):
        batch_size, seq_len = x.size()
        x = x.to(device)
        padding_mask = padding_mask.to(device) #.bool()
        look_ahead_mask = look_ahead_mask.to(device)

        token_emb = self.token_embedding(x)
        pos_emb = self.positional_encoding[:seq_len].unsqueeze(0).to(device)
        emb = token_emb + pos_emb

        emb = emb.transpose(0, 1)
        dummy_memory = torch.zeros(1, batch_size, emb.size(-1), device=device)

        out = self.decoder(tgt=emb, memory=dummy_memory, tgt_mask=look_ahead_mask, tgt_key_padding_mask=padding_mask)
        out = out.transpose(0, 1)
        return self.output_layer(out)


def generate_smiles(model, dataset, start_token, max_len=100, device=None, k=10):
    model.eval()
    x = torch.tensor([[start_token]], dtype=torch.int64).to(device)
    generated_sequence = [start_token]

    idx_to_token = {v: k for k, v in dataset.dictionary.items()}
    eos_token_id = dataset.dictionary['EOS']

    for _ in range(max_len - 1):
        seq_len = x.size(1)

        pad_mask = torch.zeros_like(x, dtype=torch.int64).to(device)
        look_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

        with torch.no_grad():
            logits = model(x, pad_mask, look_mask, device)

        last_token_logits = logits[:, -1, :].squeeze(0)
       
        topk_logits, topk_indices = torch.topk(last_token_logits, k)
        temperature = 1.2
        topk_logits = topk_logits / temperature

        probs = torch.softmax(topk_logits, dim=-1)
        sample_idx = torch.multinomial(probs, num_samples=1).item()
        next_token = topk_indices[sample_idx].item()
        
        if next_token == eos_token_id:
            break

        generated_sequence.append(next_token)
        x = torch.cat([x, torch.tensor([[next_token]], dtype=torch.int64).to(device)], dim=1)

    return ''.join([idx_to_token[idx] for idx in generated_sequence])



if __name__ == "__main__":
    path_to_smiles = os.path.join(".", "smiles_train.txt")
    dataset = SmilesDataset(path_to_smiles)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmilesTransformerDecoder(vocab_size=dataset.vocab_size, max_seq_length=101).to(device)

    num_epochs = 3
    lr = 1e-4 
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    PAD_IDX = dataset.dictionary['PAD']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    losses = []
    model.train()
    for epoch in range(num_epochs):
        batch_loss = []
        for x, pad_mask in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            pad_mask = pad_mask.to(device)

            x_in = x[:, :-1] 
            y = x[:, 1:]
            pad_in = pad_mask[:, :-1]     
            seq_len = x_in.size(1)
            look_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

            logits = model(x_in, pad_in, loolsk_mask, device)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()

            batch_loss.append(loss.item())

        avg_loss = sum(batch_loss) / len(batch_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.5f}")
        losses.append(avg_loss)
        
        for i in range(20):
            generated_smiles = generate_smiles(model, dataset, start_token=dataset.dictionary['SOS'], device=device, k=5)
            print("Generated SMILES:", generated_smiles[3:])    # Do not print SOS Token

