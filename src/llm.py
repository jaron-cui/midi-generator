# head
# - embedding key, query, value: key * transposed(query swapping T and C dim), wei softmax
# - linear layers on key query value?
# - attention * value
# feed forward
# - linear layers with relu, layer norm, dropout
# residuals - after head and feed forward
import typing

import torch
import torch.nn as nn

import miditok
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path

import time
from pygame import mixer

# with open('../input.txt', 'r', encoding='utf-8') as f:
#   text = f.read()

mixer.init()

vocab_size = 30000
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 5
n_embd = 128
n_head = 8
n_layer = 4
dropout = 0.2
# create a mapping from characters to integers
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
# Train and test splits
# data = torch.tensor(encode(text), dtype=torch.long)
# n = int(0.9*len(data)) # first 90% will be train, rest val
# train_data = data[:n]
# val_data = data[n:]

print('making tokenizer')

## Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(params=Path('model_cache/tokenizer.json'))

print('training tokenizer')

# Train the tokenizer with Byte Pair Encoding (BPE)
files_paths = list(Path('C:/Users/clack/Downloads/midis').glob("**/*.mid"))
# tokenizer.train(vocab_size=vocab_size, files_paths=files_paths)
# tokenizer.save_pretrained(Path('model_cache'))
tokenizer.from_pretrained(Path('model_cache'))
# And pushing it to the Hugging Face hub (you can download it back with .from_pretrained)
# tokenizer.push_to_hub("username/model-name", private=True, token="your_hf_token")

print('splitting midi files')

# Split MIDIs into smaller chunks for training
dataset_chunks_dir = Path('training_data/chunks')
#split_files_for_training(
#    files_paths=files_paths,
#    tokenizer=tokenizer,
#    save_dir=dataset_chunks_dir,
#    max_seq_len=block_size,
#)

midi_file_paths = list(dataset_chunks_dir.glob("**/*.mid"))
n = int(0.9*len(midi_file_paths)) # first 90% will be train, rest val

print('creating train and test datasets')

# Create a Dataset, a DataLoader and a collator to train a model
train_dataset = DatasetMIDI(
    files_paths=midi_file_paths[:n],
    tokenizer=tokenizer,
    max_seq_len=block_size,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)

train_collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True, shift_labels=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_collator, shuffle=True, pin_memory=True)


test_dataset = DatasetMIDI(
    files_paths=midi_file_paths[n:],
    tokenizer=tokenizer,
    max_seq_len=block_size,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
test_collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True, shift_labels=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_collator, pin_memory=True)

print('done creating datasets')


class Head(nn.Module):
  def __init__(self, embed_dim: int, head_size: int):
    super().__init__()
    self.head_size = head_size
    self.key = nn.Linear(embed_dim, head_size, bias=False)
    self.query = nn.Linear(embed_dim, head_size, bias=False)
    self.value = nn.Linear(embed_dim, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, attn_mask: typing.Union[torch.Tensor, None]) -> torch.Tensor:
    b, t, c = x.shape

    key = self.key(x)
    query = self.query(x)
    value = self.value(x)

    # divide by square root of head size to maintain variance
    wei = query @ key.transpose(1, 2) * self.head_size**-0.5
    wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
    if attn_mask is not None:
      attn_mask = attn_mask.unsqueeze(1).expand_as(wei)
      wei = wei.masked_fill(attn_mask == 0, float('-inf'))
    wei = torch.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    return wei @ value


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads: int, embed_dim: int):
    super().__init__()
    self.heads = nn.ModuleList([Head(embed_dim, embed_dim // num_heads) for _ in range(num_heads)])
    self.proj = nn.Linear(embed_dim, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, attn_mask: typing.Union[torch.Tensor, None]):
    out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)
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
    self.sa = MultiHeadAttention(heads, embed_dim)
    self.ffwd = FeedForward(embed_dim)
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)

  def forward(self, x: torch.Tensor, attn_mask: typing.Union[torch.Tensor, None]) -> torch.Tensor:
    x = self.sa(self.ln1(x), attn_mask) + x
    x = self.ffwd(self.ln2(x)) + x
    return x


class LM(nn.Module):
  def __init__(self, embed_dim: int, vocab_size: int, block_size: int, layers: int):
    super().__init__()
    self.block_size = block_size
    self.token_embedding = nn.Embedding(vocab_size, embed_dim)
    self.position_embedding = nn.Embedding(block_size, embed_dim)
    self.blocks = nn.ModuleList([Block(embed_dim, n_head) for _ in range(layers)])
    self.ln = nn.LayerNorm(embed_dim)
    self.lm_head = nn.Linear(embed_dim, vocab_size)

  def forward(self, x: torch.Tensor, targets=None, attn_mask: torch.Tensor = None):
    b, t = x.shape

    token = self.token_embedding(x)
    position = self.position_embedding(torch.arange(t, device=device))

    x = token + position
    for block in self.blocks:
      x = block(x, attn_mask)
    x = self.ln(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
      b, t, c = logits.shape
      logits = logits.view(b * t, c)
      targets = targets.view(b * t)
      loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
      attn_mask = attn_mask.view(b * t)
      loss = loss * attn_mask
      loss = loss.sum() / attn_mask.sum()

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


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for (k, batch) in enumerate(test_dataloader):
      if k >= eval_iters:
        break
      X, Y, attn = batch['input_ids'], batch['labels'], batch['attention_mask']
      X, Y, attn = X.to(device), Y.to(device), attn.to(device)
      logits, loss = model(X, Y, attn)
      losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


# data loading
# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y


model = LM(embed_dim=n_embd, vocab_size=vocab_size, block_size=block_size, layers=n_layer)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def train(max_iters: int):
  losses = torch.zeros(eval_iters)
  k = 0
  step = 0
  while step < max_iters:
    for batch in train_dataloader:
      # x, y, attn = batch['input_ids'], batch['labels'], batch['attention_mask']
      # print('X:', x.shape, x[0])
      # print('Y', y.shape, y[0])
      # print('attn', attn.shape, attn[0])
      # print('mask?', x.masked_fill(attn == 0, -1))
      # return
      if step >= max_iters:
        break

      # every once in a while evaluate the loss on train and val sets
      if step % eval_interval == 0 or step == max_iters - 1:
        test_loss = estimate_loss()
        print(f"step {step}: train loss {losses[:k].mean().item():.4f}, val loss {test_loss:.4f}")

      # evaluate the loss
      x, y, attn = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)
      logits, loss = model(x, y, attn)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

      if k < len(losses):
        losses[k] = loss
      else:
        k = 0
      k += 1


def handle_command(parts: list[str]):
  command = parts[0]
  if command == 'load':
    model.load_state_dict(torch.load('model_cache/weights.pth', weights_only=True))
    print('Loaded model!')
  elif command == 'train':
    iter = 5000 if len(parts) == 1 else int(parts[1])
    print(f'Training for {iter} iterations...')
    train(iter)
  elif command == 'save':
    torch.save(model.state_dict(), 'model_cache/weights.pth')
    print('Saved model!')
  elif command == 'generate':
    length = 2000 if len(parts) == 1 else int(parts[1])
    # generate from the model
    print('Generating....')
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    tokens = m.generate(context, max_new_tokens=length)[0].tolist()
    score = tokenizer.decode(tokens)
    score.dump_midi('model_cache/generated.mid')
  elif command == 'play':
    try:
      mixer.music.load('model_cache/generated.mid')
      print('Playing music!')
    except FileNotFoundError:
      print('No generated music!')
      return
    mixer.music.play()
    while mixer.music.get_busy():
      time.sleep(1)
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

