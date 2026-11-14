import math

import torch
from torch import nn


torch.manual_seed(42)

text = "the cat sleeps."
chars = sorted(set(text))
stoi = {char: i for i, char in enumerate(chars)}
tokens = torch.tensor([stoi[char] for char in text], dtype=torch.long)
inputs = tokens[:-1]
targets = tokens[1:]


class CharLanguageModel(nn.Module):
    def __init__(self, vocab_size, sequence_length, embedding_size=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(sequence_length, embedding_size)
        self.query = nn.Linear(embedding_size, embedding_size, bias=False)
        self.key = nn.Linear(embedding_size, embedding_size, bias=False)
        self.value = nn.Linear(embedding_size, embedding_size, bias=False)
        self.output = nn.Linear(embedding_size, vocab_size)

    def forward(self, token_ids):
        sequence_length = token_ids.size(0)
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)

        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        scores = (query @ key.T) / math.sqrt(query.size(-1))

        future_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=token_ids.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        return self.output(attention @ value)


model = CharLanguageModel(len(chars), len(inputs))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

logits = model(inputs)
loss = loss_function(logits, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")
