import datetime
import os
import random
import time
from typing import Callable

from synthetic_music_generation.generate import generate_piece


def generate_dataset(
    directory: str,
    num_files: int,
    file_namer: Callable[[int], str] = None,
    seed: int = None,
    log_frequency: int = 10
):
  """
  Generate a synthetic dataset in a provided directory.

  Example usage:
  ::
    generate_dataset('./dataset', 100)

  :param directory: the directory in which synthetic MIDI files should be saved
  :param num_files: the number of files the dataset should constitute
  :param file_namer: optionally, a function that takes in a file index and formats a custom file name for that index
  :param seed: seed for randomly generating the dataset
  :param log_frequency: the ETA will be calculated and printed every *log_frequency* files
  :return: None
  """
  # set default file namer to name files "synthetic_midi0023.mid"
  if file_namer is None:
    def file_namer(i):
      return f'synth{i:04}.mid'
  # create the save directory
  os.makedirs(directory, exist_ok=True)

  # generate files until the number of tokens is at least target_num_tokens
  print(f'Generating synthetic dataset of {num_files} files in \'{directory}\'. Time: {datetime.datetime.now()}')
  random.seed(seed)
  start_time = time.time()
  for file_index in range(num_files):
    if file_index % log_frequency == 0:
      if file_index > 0:
        time_elapsed = time.time() - start_time
        eta = datetime.datetime.now() + datetime.timedelta(seconds=(time_elapsed / file_index) * (num_files - file_index))
      else:
        eta = 'N/A'
      print(f' - Generating file {file_index}. ETA: {eta}')
    file_name = file_namer(file_index)
    file_path = os.path.join(directory, file_name)

    # generate the next synthetic MIDI file at file_path
    generate_piece(file_path, merge_tracks=False, left_hand=False)

    file_index += 1

  print(f'Generated {num_files} files. Time: {datetime.datetime.now()}')


# generate_dataset('./data', num_files=700, seed=0)
