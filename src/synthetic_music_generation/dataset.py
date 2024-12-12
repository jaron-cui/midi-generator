import datetime
import os
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np

from synthetic_music_generation.generate import generate_piece


def generate_dataset(
    directory: str,
    target_num_tokens: int,
    tokenizer: Callable[[str], np.ndarray],
    file_namer: Callable[[int], str] = None,
    seed: int = None
):
  """
  Generate a synthetic dataset in a provided directory.

  Example usage:
  ::
    directory = './dataset'
    target_num_tokens = 20000
    def tokenizer(midi_file_path: str) -> np.ndarray:
      ...
    generate_dataset(directory, target_num_tokens, tokenizer

  :param directory: the directory in which synthetic MIDI files should be saved
  :param target_num_tokens: the number of tokens the dataset should constitute
  :param tokenizer: a function which takes in a single MIDI file path and returns its tokenized contents
  :param file_namer: optionally, a function that takes in a file index and formats a custom file name for that index
  :param seed: seed for randomly generating the dataset
  :return: None
  """
  # set default file namer to name files "synthetic_midi0023.mid"
  if file_namer is None:
    def file_namer(i):
      return f'synth{i:04}.mid'
  # create the save directory
  os.makedirs(directory, exist_ok=True)

  # generate files until the number of tokens is at least target_num_tokens
  print(f'Generating synthetic dataset of >= {target_num_tokens} tokens.')
  random.seed(seed)
  start_time = time.time()
  num_tokens_generated = 0
  file_index = 0
  while num_tokens_generated < target_num_tokens:
    if file_index % 10 == 0:
      if num_tokens_generated > 0:
        time_elapsed = time.time() - start_time
        eta = datetime.datetime.now() + datetime.timedelta(seconds=(time_elapsed / num_tokens_generated) * target_num_tokens)
      else:
        eta = 'N/A'
      print(f' - Generating file {file_index}. ETA: {eta}')
    file_name = file_namer(file_index)
    file_path = os.path.join(directory, file_name)

    # generate the next synthetic MIDI file at file_path
    generate_piece(file_path)

    # tokenize and record the size of the generated file
    tokens = tokenizer(file_path)
    num_tokens = tokens.size

    num_tokens_generated += num_tokens
    file_index += 1

  print(f'Generated {file_index} files representing {num_tokens_generated} tokens.')


# generate_dataset('./data', target_num_tokens=10, tokenizer=lambda path: np.zeros(1), seed=0)
