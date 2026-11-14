import torch
from torch import nn

# 固定随机种子，方便重复实验
torch.manual_seed(42)

# ===== 1. Token：文字变成编号 =====

text = "the cat sleeps."

# 收集不同字符，建立字符词表
chars = sorted(set(text))

# 字符 → 编号
stoi = {char: i for i, char in enumerate(chars)}

# 编号 → 字符
itos = {i: char for char, i in stoi.items()}

tokens = [stoi[char] for char in text]
decoded = "".join(itos[token] for token in tokens)

print("词表:", stoi)
print("Token编号:", tokens)
print("还原文字:", decoded)

# ===== 2. 准备输入和正确答案 =====

inputs = tokens[:-1]
targets = tokens[1:]

print("输入文字:", "".join(itos[i] for i in inputs))
print("目标文字:", "".join(itos[i] for i in targets))
print("第一个位置的正确答案:", itos[inputs[0]], "→", itos[targets[0]])

# ===== 3. Embedding：编号变成向量 =====

x = torch.tensor(inputs, dtype=torch.long)

# 每个字符对应8个数字
embedding = nn.Embedding(len(chars), 8)
vectors = embedding(x)

print("向量形状:", vectors.shape)
print("第一个字符的向量:", vectors[0].detach())

# ===== 4. 加入位置信息 =====

positions = torch.arange(len(inputs))
position_embedding = nn.Embedding(len(inputs), 8)

hidden = vectors + position_embedding(positions)

print("加入位置后的形状:", hidden.shape)

# ===== 5. 单头因果 Attention =====

query = nn.Linear(8, 8, bias=False)
key = nn.Linear(8, 8, bias=False)
value = nn.Linear(8, 8, bias=False)

q = query(hidden)
k = key(hidden)
v = value(hidden)

# 计算各位置之间的参考分数
scores = (q @ k.T) / (8 ** 0.5)

# 遮住未来位置，防止偷看答案
mask = torch.triu(
    torch.ones(len(inputs), len(inputs), dtype=torch.bool),
    diagonal=1
)

scores = scores.masked_fill(mask, float("-inf"))

# 分数转换为权重
weights = torch.softmax(scores, dim=-1)

# 按权重汇总信息
attention_output = weights @ v

print("Attention形状:", attention_output.shape)
print("第一个位置的权重:", weights[0].detach())

# ===== Task A: Multi-Step Training =====

import torch
from torch import nn

torch.manual_seed(42)

text_data = "the cat sleeps. the cat eats. " * 30

vocab = sorted(set(text_data))
char_to_id = {c: i for i, c in enumerate(vocab)}

data = torch.tensor(
    [char_to_id[c] for c in text_data],
    dtype=torch.long
)

x_train = data[:-1]
y_train = data[1:]

# Simple baseline model
baseline = nn.Sequential(
    nn.Embedding(len(vocab), 16),
    nn.Linear(16, len(vocab))
)

optimizer_baseline = torch.optim.Adam(
    baseline.parameters(),
    lr=0.01
)

for step in range(201):
    logits = baseline(x_train)

    loss = nn.functional.cross_entropy(
        logits,
        y_train
    )

    optimizer_baseline.zero_grad()
    loss.backward()
    optimizer_baseline.step()

    if step % 20 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

# ===== 6. Predict next character =====

vocab_size = len(chars)

output_layer = nn.Linear(8, vocab_size)

logits = output_layer(attention_output)

print("Logits shape:", logits.shape)

# ===== 7. Transformer Loss =====

target_tensor = torch.tensor(targets, dtype=torch.long)

transformer_loss = nn.functional.cross_entropy(
    logits,
    target_tensor
)

print("Transformer Loss:", transformer_loss.item())

# ===== 8. Train the actual Transformer =====

# Create these layers ONCE, not inside the training loop.
mlp = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 8)
)

attention_projection = nn.Linear(8, 8)
norm1 = nn.LayerNorm(8)
norm2 = nn.LayerNorm(8)

# Include ALL Transformer layers in the optimizer.
# These are separate from your earlier baseline model.
transformer_layers = nn.ModuleList([
    embedding, position_embedding,
    query, key, value,
    attention_projection,
    norm1, norm2, mlp, output_layer
])

optimizer_transformer = torch.optim.Adam(
    transformer_layers.parameters(),
    lr=0.01
)


def transformer_forward(token_ids):
    # Recalculate embeddings using the CURRENT parameters.
    length = len(token_ids)
    positions = torch.arange(length, device=token_ids.device)

    h = embedding(token_ids) + position_embedding(positions)

    # Attention: Query, Key, Value.
    q = query(h)
    k = key(h)
    v = value(h)

    scores = (q @ k.T) / (8 ** 0.5)

    # Block future characters.
    future_mask = torch.triu(
        torch.ones(
            length, length,
            dtype=torch.bool,
            device=token_ids.device
        ),
        diagonal=1
    )

    scores = scores.masked_fill(future_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    # Attention, then MLP, with residual connections and normalization.
    h = norm1(h + attention_projection(weights @ v))
    h = norm2(h + mlp(h))

    # One prediction score per vocabulary character.
    return output_layer(h)


# Use your ORIGINAL Transformer inputs and targets.
transformer_x = torch.tensor(inputs, dtype=torch.long)
transformer_y = torch.tensor(targets, dtype=torch.long)

transformer_layers.train()

for step in range(201):
    # IMPORTANT: build a fresh forward computation every time.
    new_logits = transformer_forward(transformer_x)

    train_loss = nn.functional.cross_entropy(
        new_logits,
        transformer_y
    )

    if not torch.isfinite(train_loss):
        raise RuntimeError("Loss became NaN or infinity. Stop training.")

    if step % 20 == 0:
        print(
            f"Transformer step {step}: "
            f"loss = {train_loss.item():.4f}"
        )

    # Step 0: before training. Step 200: after 200 updates.
    if step < 200:
        optimizer_transformer.zero_grad()
        train_loss.backward()
        optimizer_transformer.step()

    # ===== 9. Generate text using the trained Transformer =====

transformer_layers.eval()

prompt = "the "
generated_ids = [stoi[character] for character in prompt]

# Stay within the position embedding's supported input length.
max_input_length = position_embedding.num_embeddings

with torch.no_grad():
    while len(generated_ids) <= max_input_length:
        input_ids = torch.tensor(
            generated_ids,
            dtype=torch.long,
            device=embedding.weight.device
        )

        prediction_scores = transformer_forward(input_ids)

        # Use the LAST position to predict the NEXT character.
        next_id = prediction_scores[-1].argmax().item()
        generated_ids.append(next_id)

        # Stop at the end of a sentence.
        if itos[next_id] == ".":
            break

generated_text = "".join(itos[token] for token in generated_ids)

print("Prompt:", prompt)
print("Generated:", generated_text)

# ===== 10. Inference Optimization =====

import time
import statistics

def benchmark(mode):
    times = []

    for trial in range(3):
        start = time.perf_counter()

        with mode():
            for _ in range(100):
                transformer_forward(transformer_x)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return statistics.median(times)


# Warm-up
with torch.inference_mode():
    for _ in range(10):
        transformer_forward(transformer_x)

normal = benchmark(torch.enable_grad)
optimized = benchmark(torch.inference_mode)

with torch.no_grad():
    output_normal = transformer_forward(transformer_x)

with torch.inference_mode():
    output_optimized = transformer_forward(transformer_x)

print("Normal:", normal)
print("Optimized:", optimized)
print("Speedup:", normal / optimized)
print(
    "Predictions match:",
    torch.allclose(output_normal, output_optimized, atol=1e-6)
)