# head
# - embedding key, query, value: key * transposed(query swapping T and C dim), wei softmax
# - linear layers on key query value?
# - attention * value
# feed forward
# - linear layers with relu, layer norm, dropout
# residuals - after head and feed forward


import torch
import torch.nn as nn


class Head(nn.Module):
  def __init__(self, embed_dim: int, head_size: int):
    super().__init__()
    self.head_size = head_size
    self.key = nn.Linear(embed_dim, head_size, bias=False)
    self.query = nn.Linear(embed_dim, head_size, bias=False)
    self.value = nn.Linear(embed_dim, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor):
    b, t, c = x.shape

    key = self.key(x)
    query = self.query(x)
    value = self.value(x)

    # divide by square root of head size to maintain variance
    wei = query @ key.transpose(1, 2) * self.head_size**-0.5
    wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
    wei = torch.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    return wei @ value


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads: int, head_size: int, embed_dim: int):
    super().__init__()
    self.heads = nn.ModuleList([Head(embed_dim, embed_dim // num_heads) for _ in range(num_heads)])
    self.proj = nn.Linear(embed_dim, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out


class FeedForward(nn.Module):
  def __init__(self, embed_dim: int):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(embed_dim, 4 * embed_dim),
      nn.ReLU(),
      nn.Linear(4 * embed_dim, embed_dim),
      nn.Dropout(dropout)
    )

  def forward(self, x: torch.Tensor):
    return self.layers(x)


class Block(nn.Module):
  def __init__(self, embed_dim: int, heads: int):
    super().__init__()
    self.sa = MultiHeadAttention(heads, embed_dim // heads, embed_dim)
    self.ffwd = FeedForward(embed_dim)
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)

  def forward(self, x: torch.Tensor):
    x = self.sa(self.ln1(x)) + x
    x = self.ffwd(self.ln2(x)) + x
    return x


class LM(nn.Module):
  def __init__(self, embed_dim: int, vocab_size: int, block_size: int, layers: int):
    super().__init__()
    self.block_size = block_size
    self.token_embedding = nn.Embedding(vocab_size, embed_dim)
    self.position_embedding = nn.Embedding(block_size, embed_dim)
    self.blocks = nn.Sequential(*[Block(embed_dim, n_head) for _ in range(layers)])
    self.ln = nn.LayerNorm(embed_dim)
    self.lm_head = nn.Linear(embed_dim, vocab_size)

  def forward(self, x: torch.Tensor, targets=None):
    b, t = x.shape

    token = self.token_embedding(x)
    position = self.position_embedding(torch.arange(t, device=device))
    x = token + position
    x = self.blocks(x)
    x = self.ln(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
      b, t, c = logits.shape
      logits = logits.view(b * t, c)
      targets = targets.view(b * t)
      loss = torch.nn.functional.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, x: torch.Tensor, max_new_tokens: int):
    for _ in range(max_new_tokens):
      context = x[:, -self.block_size:]
      logits, __ = self(context)
      logits = logits[:, -1, :]
      probs = torch.softmax(logits, dim=-1)
      x_next = torch.multinomial(probs, num_samples=1)
      x = torch.cat([x, x_next], dim=1)
    return x


with open('../input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 8
n_layer = 4
dropout = 0.2
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


model = LM(embed_dim=n_embd, vocab_size=vocab_size, block_size=block_size, layers=n_layer)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def train(max_iters: int):
  for iter in range(max_iters):

      # every once in a while evaluate the loss on train and val sets
      if iter % eval_interval == 0 or iter == max_iters - 1:
          losses = estimate_loss()
          print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

      # sample a batch of data
      xb, yb = get_batch('train')

      # evaluate the loss
      logits, loss = model(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()


def handle_command(parts: list[str]):
  command = parts[0]
  if command == 'load':
    model.load_state_dict(torch.load('../model_cache/weights.pth', weights_only=True))
    print('Loaded model!')
  elif command == 'train':
    iter = 5000 if len(parts) == 1 else int(parts[1])
    print(f'Training for {iter} iterations...')
    train(iter)
  elif command == 'save':
    torch.save(model.state_dict(), '../model_cache/weights.pth')
    print('Saved model!')
  elif command == 'generate':
    length = 2000 if len(parts) == 1 else int(parts[1])
    # generate from the model
    print('Generating....')
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=length)[0].tolist()))
  else:
    print('Unknown command!')


while True:
  text_input = input('Enter command (load, train <iter=5000>, generate <len=2000>, save, exit):')
  if len(text_input) == 0:
    continue
  parts = text_input.split(' ')
  if len(parts) == 0:
    continue
  if parts[0] == 'exit':
    break
  handle_command(parts)

