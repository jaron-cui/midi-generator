import datetime
from collections import Counter
import pickle
import glob
from typing import List

import utils
import os
import numpy as np

from synthetic_music_generation.dataset import generate_dataset


def extract_events(input_path, chord=False):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    if chord:
        chord_items = utils.extract_chords(note_items)
        items = chord_items + tempo_items + note_items
    else:
        items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events


def create_dictionary(midi_paths: List[str], dictionary_path: str):
    counts = Counter()
    # all_elements = []
    # for midi_file in glob.glob("D:/files_2_jaron/midi_webscrape/*.mid*", recursive=True):
    for i, midi_file in enumerate(midi_paths):
        if i % 100 == 0:
            print(f'Added {i} files to dictionary so far. Time: {datetime.datetime.now()}')
        elements = []
        try:
            events = extract_events(midi_file) # If you're analyzing chords, use `extract_events(midi_file, chord=True)`
            for event in events:
                element = '{}_{}'.format(event.name, event.value)
                elements.append(element)

        except:
            print("Couldn't evaluate, so removing", str(midi_file))
            os.remove(str(midi_file))
        counts.update(elements)

    # counts = Counter(all_elements)
    event2word = {c: i for i, c in enumerate(counts.keys())}
    word2event = {i: c for i, c in enumerate(counts.keys())}
    pickle.dump((event2word, word2event), open(dictionary_path, 'wb'))


# model settings
x_len = 512
mem_len = 512
n_layer = 12
d_embed = 512
d_model = 512


def prepare_data(midi_paths, dictionary_path: str, max_tokens: int = None) -> np.ndarray:
    event2word, word2event = pickle.load(open(dictionary_path, 'rb'))
    # extract events
    all_events = []
    for path in midi_paths:
        events = extract_events(path)
        all_events.append(events)
    # event to word
    all_words = []
    for events in all_events:
        words = []
        for event in events:
            e = '{}_{}'.format(event.name, event.value)
            if e in event2word:
                words.append(event2word[e])
            else:
                # OOV
                if event.name == 'Note Velocity':
                    # replace with max velocity based on our training data
                    words.append(event2word['Note Velocity_21'])
                else:
                    # something is wrong
                    # you should handle it for your own purpose
                    print('something is wrong! {}'.format(e))
        all_words.append(words)
    # to training data
    group_size = 5
    segments = []
    for words in all_words:
        pairs = []
        for i in range(0, len(words) - x_len - 1, x_len):
            x = words[i:i + x_len]
            y = words[i + 1:i + x_len + 1]
            pairs.append([x, y])
        pairs = np.array(pairs)
        # abandon the last
        for i in np.arange(0, len(pairs) - group_size, group_size * 2):
            data = pairs[i:i + group_size]
            if len(data) == group_size:
                segments.append(data)
        if max_tokens is not None and len(segments) >= max_tokens // (2 * 5 * 512):
            break
    if max_tokens is None:
        return np.array(segments)
    max_segments = max_tokens // (2 * 5 * 512)
    segments = np.array(segments)[:max_segments]
    return segments


def create_synthetic_dict(epochs: int, dictionary_path: str):
    midi_paths = glob.glob('../../output/real_data/real_data_1hand/*.mid*', recursive=True)
    # generate unique dataset for each epoch
    for i in range(epochs):
        if i not in range(28):
            generate_dataset(f'../../output/synthetic_data/synth_data_1hand/epoch{i}', 750, seed=i, log_frequency=50)
        midi_paths += glob.glob(f'../../output/synthetic_data/synth_data_1hand/epoch{i}/*.mid*', recursive=True)
    # create the dictionary from the real data and all synthetic data
    create_dictionary(midi_paths, dictionary_path)

# midi_paths = glob.glob('synthetic_music_generation/synth_data/epoch4/*.mid*', recursive=True)
# create_dictionary(midi_paths, 'temp_dictionary.pkl')
# midi_paths = glob.glob('synthetic_music_generation/synth_data/epoch4/*.mid*', recursive=True)
# print(prepare_data(midi_paths, 'temp_dictionary.pkl').size)


create_synthetic_dict(40, '../../output/pickled_dictionaries/real+synth_dictionary_1hand.pkl')
#396

# pickling
for epoch in range(40):
    midi_paths = glob.glob(f'../../output/synthetic_data/synth_data_1hand/epoch{epoch}/*.mid*', recursive=True)
    processed_data = prepare_data(midi_paths, '../../output/pickled_dictionaries/real+synth_dictionary_1hand.pkl', max_tokens=1884160)
    save_path = f'../../output/synthetic_data/synth_data_1hand/pickles/epoch{epoch}.npy'
    np.save(save_path, processed_data)
    print(f'Processed epoch {epoch} into {processed_data.size} tokens, saved at {save_path}. Time: {datetime.datetime.now()}')
