import math
import random
from itertools import combinations, combinations_with_replacement
from typing import Union, List, TypeVar, Callable, Tuple

from midiutil import MIDIFile
from mido import MidiFile, merge_tracks

from synthetic_music_generation.representations import parse_note, Chord, Note, SectionCharacteristics

T = TypeVar('T')


def item_at_time(sequence: List[T], t: float) -> T:
  current_time = 0
  for element in sequence:
    if current_time + element.duration >= t:
      return element
    current_time += element.duration
  raise ValueError(f'There is no item at time {t} in sequence {sequence} of total time {current_time}')


def change_key(notes: List[Note], src_chord_progression: List[Chord], tgt_chord_progression: List[Chord]) -> List[Note]:
  new_notes = []
  current_time = 0
  for note in notes:
    src_chord = item_at_time(src_chord_progression, current_time)
    tgt_chord = item_at_time(tgt_chord_progression, current_time)
    scale_indices = {src_chord.key.pitch_as_scale_note(pitch) for pitch in note.pitches}
    adapted_pitches = {tgt_chord.key.pitch(index) for index in scale_indices}
    new_notes.append(Note(adapted_pitches, note.duration))
  return new_notes


def calculate_interval(note1: Union[int, str], note2: Union[int, str]) -> float:
  index1, offset1 = parse_note(note1)
  index2, offset2 = parse_note(note2)
  return (index2 + offset2 / 2) - (index1 + offset1 / 2)


def create_notes(
    pitches: List[int], rhythm: List[float], chord_progression: List[Chord], section_spec: SectionCharacteristics) -> \
    List[Note]:
  # longer notes are more likely to be harmonized
  # notes at the beginnings and ends of sequences are more likely to be harmonized
  # the 'center of mass' of sequential harmonization chords like to stay close until they can't -
  # harmonizations have a limited spread based on hand width
  notes = []
  recent_harmony_pitches = set()
  current_time = 0
  for pitch, duration in zip(pitches, rhythm):
    # chance of note not being harmonized
    if random.random() < math.pow(1 - section_spec.beat_harmonization_rate, duration):
      notes.append(Note({pitch}, duration))
      continue
    chord = item_at_time(chord_progression, current_time)
    # collect the chord pitches within reach as possible harmony pitches
    possible_harmony_pitches = set()
    for direction in [-1, 1]:
      next_pitch = next_chord_pitch(pitch, chord, direction)
      while abs(next_pitch - pitch) <= section_spec.max_pitch_interval:
        if abs(next_pitch - pitch) > 2 and -12 <= next_pitch <= 20:
          possible_harmony_pitches.add(next_pitch)
        next_pitch = next_chord_pitch(next_pitch, chord, direction)
    min_degree, max_degree = section_spec.harmonization_bounds
    degree = round(random.random() * (max_degree - min_degree) + min_degree)
    if degree > 0:
      possible_harmonizations = [
        h for h in combinations(possible_harmony_pitches, degree) if max(h) - min(h) <= section_spec.max_pitch_interval]
      random.shuffle(possible_harmonizations)
      # sort by the sum of the distances from each pitch to its nearest pitch in recent_harmony_pitches
      # this is intended to keep sequential harmonizing chords within a similar vicinity to one another
      possible_harmonizations.sort(
        key=lambda h: sum(min([abs(p - q) for q in recent_harmony_pitches] + [0]) for p in h))
    else:
      possible_harmonizations = []
    harmonization = possible_harmonizations[0] if possible_harmonizations else []
    notes.append(Note(set(harmonization).union([pitch]), duration))
    current_time += duration
  return notes


def chord_excerpt(chord_progression: List[Chord], start_time: float, end_time: float) -> List[Chord]:
  chord_progression = [chord.copy() for chord in chord_progression]
  start_index, start_clip = None, None
  end_index, end_clip = None, None
  cumulative_time = 0
  for index in range(len(chord_progression)):
    chord_duration = chord_progression[index].duration
    if start_index is None and cumulative_time + chord_duration > start_time:
      start_index = index
      start_clip = start_time - cumulative_time
    if end_index is None and cumulative_time >= end_time:
      end_index = index
      end_clip = cumulative_time + chord_duration - end_time
      break
    cumulative_time += chord_duration
  if start_index is None:
    return []
  if end_index is None:
    end_index = len(chord_progression) - 1
    end_clip = 0
  chord_subsequence = chord_progression[start_index:end_index + 1]
  chord_subsequence[0].duration -= start_clip
  chord_subsequence[-1].duration -= end_clip
  if chord_subsequence[0].duration <= 0:
    chord_subsequence = chord_subsequence[1:]
  return chord_subsequence


#
def generate_arpeggio_possibilities(start_pitch: int, end_pitch: int, chord_progression: List[Chord],
                                    include_end_pitch: bool, spec: SectionCharacteristics):
  return generate_oscillating_sequences(start_pitch, end_pitch, chord_progression, include_end_pitch, next_chord_pitch,
                                        spec)


def generate_run_possibilities(start_pitch: int, end_pitch: int, chord_progression: List[Chord],
                               include_end_pitch: bool, spec: SectionCharacteristics):
  return generate_oscillating_sequences(start_pitch, end_pitch, chord_progression, include_end_pitch, next_scale_pitch,
                                        spec)


def next_chord_pitch(pitch: int, chord: Chord, direction: int) -> int:
  chord_pitches = [chord.key.pitch(index) for index in chord.notes]
  chord_pitches = [p - 12 for p in chord_pitches] + chord_pitches + [chord_pitches[0] + 12]
  chord_pitches.sort(reverse=direction < 0)

  pitch_octave = pitch // 12
  normalized_pitch = pitch - pitch_octave * 12
  for chord_pitch in chord_pitches:
    if (direction * chord_pitch) > (direction * normalized_pitch):
      return chord_pitch + pitch_octave * 12


def next_scale_pitch(pitch: int, chord: Chord, direction: int) -> int:
  chord_pitches = chord.key.pitches
  chord_pitches = [chord_pitches[-1] - 12] + chord_pitches + [chord_pitches[0] + 12]
  chord_pitches.sort(reverse=direction < 0)

  pitch_octave = pitch // 12
  normalized_pitch = pitch - pitch_octave * 12
  for chord_pitch in chord_pitches:
    if (direction * chord_pitch) > (direction * normalized_pitch):
      return chord_pitch + pitch_octave * 12


def generate_oscillating_sequences(
    start_pitch: int,
    end_pitch: int,
    chord_progression: List[Chord],
    include_end_pitch: bool,
    interval_navigator: Callable[[int, Chord, int], int],
    spec: SectionCharacteristics = None
) -> List[List[Tuple[List[int], List[float]]]]:
  """
  Generates rhythmic sequences of notes split into monotonically pitch increasing
  or decreasing subsequences that match the given chord progression.

  :param start_pitch: The pitch of the note that the sequence will start with.
  :param end_pitch: The pitch of the note that the sequence aims to end with.
  :param chord_progression: The chord progression delineating allowed notes and sequence duration.
  :param include_end_pitch: Whether *end_pitch* should be included at the end.
  :param interval_navigator: (pitch, chord, direction) -> next_pitch; used to find the next allowed
  pitch in a given direction.
  :param spec:
  :return: A list of possible sequences split into subsequences represented as (pitches, rhythms).
  """
  # TODO: rename variables to reflect abstraction away from just arpeggios
  duration = sum(c.duration for c in chord_progression)
  # TODO: make min note duration a config
  if spec is None:
    max_num_notes = duration // 0.5
    min_num_notes = min(3, max_num_notes)
    velocity = (max_num_notes + min_num_notes) / 2
    velocity_tolerance = (max_num_notes - min_num_notes) / 2
    rhythm = generate_rhythm(duration, 0.5, 4, 0.2, velocity, velocity_tolerance, measure_duration=duration)
  else:
    min_duration, max_duration = spec.note_duration_bounds
    # max_num_notes = duration // min_duration
    # min_num_notes = min(3, max_num_notes)
    if include_end_pitch:
      min_num_notes = 3
    else:
      min_num_notes = 2
    # print(f'min note duration: {min_duration}')
    rhythm = generate_rhythm(duration, min_duration, max_duration, spec.syncopation, spec.velocity,
                             spec.velocity_tolerance, measure_duration=spec.measure_length,
                             min_note_count=min_num_notes)
  if len(rhythm) == 2 and start_pitch == end_pitch:
    return [[([start_pitch, end_pitch], rhythm)]]
  # fit_chord_notes = calculate_num_chord_notes_between(start_pitch, end_pitch, chord_progression, rhythm)
  possible_arpeggiations = []
  num_extra_peaks = 0
  while True:
    if num_extra_peaks >= len(rhythm):
      # print(f'rhythm: {rhythm}, start: {start_pitch}, end: {end_pitch}, include: {include_end_pitch}, m: {min_num_notes}')
      raise RuntimeError('Could not generate arpeggios')
    # TODO: should probably disallow directly consecutive peaks; ones right after another with no notes in-between
    for peaks in combinations(range(1, len(rhythm) - 1), num_extra_peaks):
      # subdivide into different arpeggios and try to fit
      subdivisions = [(rhythm, chord_progression)]
      index_offset = 0
      time_offset = 0
      for peak in peaks:
        section, section_chords = subdivisions[-1]
        front = section[:peak - index_offset]
        front_chords = chord_excerpt(section_chords, 0, sum(front))
        back = section[peak - index_offset:]
        back_chords = chord_excerpt(section_chords, sum(front), sum(c.duration for c in section_chords))
        subdivisions = subdivisions[:-1] + [(front, front_chords), (back, back_chords)]
        index_offset += len(front)
        time_offset += sum(front)
      # basically do another search to find any viable pitches for intermediate peaks
      heads = [([], -1, start_pitch)]
      while heads:
        arpeggios, last_subdivision, last_pitch = heads.pop()
        if last_subdivision >= len(subdivisions) - 1:
          continue
        section, section_chords = subdivisions[last_subdivision + 1]
        peak_chord = item_at_time(section_chords, sum(section[:-1]))
        include_section_end_pitch = (last_subdivision + 1 == len(subdivisions) - 1) and include_end_pitch
        if last_subdivision == len(subdivisions) - 2:
          possible_arpeggios = generate_monotonically_evolving_pitches(
            last_pitch, end_pitch, section_chords, include_section_end_pitch, section, interval_navigator)
          if len(possible_arpeggios) > 0:
            possible_arpeggiations.append(list(zip(
              arpeggios + [random.choice(possible_arpeggios)], [subdivision[0] for subdivision in subdivisions])))

        potential_peaks = []
        for direction in [-1, 1]:
          next_candidate = interval_navigator(last_pitch, peak_chord, direction)
          # 7 is the number of notes in a typical two-octave arpeggio
          # comfortably reached by a piano player with 1 crossover
          for _ in range(7 - 1):
            if -12 <= next_candidate <= 20:
              potential_peaks.append(next_candidate)
            # TODO: disqualify a candidate if it is beyond the specified note range
            next_candidate = interval_navigator(next_candidate, peak_chord, direction)
        for peak_pitch in potential_peaks:
          possible_arpeggios = generate_monotonically_evolving_pitches(
            last_pitch, peak_pitch, section_chords, include_section_end_pitch, section, interval_navigator)
          if len(possible_arpeggios) > 0:
            heads.append((arpeggios + [random.choice(possible_arpeggios)], last_subdivision + 1, peak_pitch))
    # generally, break with the minimum number of extra peaks
    if possible_arpeggiations:
      break
    num_extra_peaks += 1
  # TODO: possibly do a more sophisticated ranking and selection of arpeggios
  return possible_arpeggiations  # random.choice(possible_arpeggiations)


def generate_monotonically_evolving_pitches(
    start_pitch: int,
    end_pitch: int,
    chord_progression: List[Chord],
    include_end_pitch: bool,
    rhythm: List[float],
    interval_navigator: Callable[[int, Chord, int], int]
) -> List[List[int]]:
  """
  Generates all possible monotonically evolving pitch sequences given a function
  that informs monotonic steps.

  :param start_pitch: The pitch of the note that the sequence will start with.
  :param end_pitch: The pitch of the note that the sequence aims to end with.
  :param chord_progression: The chord progression delineating allowed notes and sequence duration.
  :param include_end_pitch: Whether *end_pitch* should be included at the end.
  :param rhythm: The rhythm that the sequence should follow.
  :param interval_navigator: (pitch, chord, direction) -> next_pitch; used to find the next allowed
  pitch in a given direction.
  :return: A list of lists of monotonically evolving pitches.
  """
  # we will assume pianists cannot do jumps greater than a tenth
  max_interval = 14

  if len(rhythm) < 2:
    return [[start_pitch] * len(rhythm)] if abs(end_pitch - start_pitch) <= max_interval else []

  arpeggios = []
  direction = math.copysign(1, end_pitch - start_pitch)

  # when the only option is to include the end pitch even though specified not to
  if not include_end_pitch and len(rhythm) == 2 and interval_navigator(start_pitch,
                                                                       item_at_time(chord_progression, rhythm[0]),
                                                                       direction) == end_pitch:
    return [[start_pitch, end_pitch]]

  # last_note_chord = item_at_time(chord_progression, sum(rhythm[:-2 if include_end_pitch else -1]))
  # last_chord_pitch = next_chord_pitch(end_pitch, last_note_chord, -direction)
  heads = [([start_pitch], 0)]
  while heads:
    pitches, rhythm_index = heads.pop()
    if include_end_pitch and rhythm_index == len(rhythm) - 1 and pitches[-1] == end_pitch:
      arpeggios.append(pitches)
    if not include_end_pitch and rhythm_index == len(rhythm) and pitches[-1] == end_pitch:
      arpeggios.append(pitches[:-1])
    if rhythm_index >= len(rhythm):
      continue

    current_time = sum(rhythm[:rhythm_index + 1])
    chord = item_at_time(chord_progression, current_time)
    next_pitch = interval_navigator(pitches[-1], chord, direction)
    while (direction * next_pitch) < (direction * end_pitch) and abs(next_pitch - pitches[-1]) <= max_interval:
      heads.append((pitches + [next_pitch], rhythm_index + 1))
      next_pitch = interval_navigator(next_pitch, chord, direction)
    if pitches[-1] != end_pitch and abs(pitches[-1] - end_pitch) <= max_interval:
      heads.append((pitches + [end_pitch], rhythm_index + 1))
  return arpeggios


def find_combinations(nums: List[float], target: float, lo: int, hi: int) -> List[List[float]]:
  result = []
  nums = list(nums)  # Convert to list if it's not already
  for length in range(max(lo, 1), hi + 1):
    for combination in combinations_with_replacement(nums, length):
      if sum(combination) == target:
        result.append(list(combination))
  return result


def count_syncopation(rhythm: List[float]) -> int:
  syncopation = 0
  current_time = 0
  for note in rhythm:
    start = current_time
    current_time += note
    end = current_time
    # could be syncopated if start or end is off-beat
    if math.floor(start) != start or math.floor(end) != end:
      # but is only syncopated if the off-beat start or end are in different beats from one another
      if not (math.ceil(start) == math.ceil(end) or math.floor(start) == math.floor(end)):
        syncopation += 1
  return syncopation


# TODO: this does not often generate triplets grouped correctly
def generate_rhythm(
    duration: float,
    min_note_length: float,
    max_note_length: float,
    syncopation_rate: float,
    velocity_per_measure: float,
    velocity_tolerance: float = 1,
    measure_duration: int = 4,
    possible_durations: List[float] = None,
    min_note_count: int = float('-inf')
) -> List[float]:
  if possible_durations is None:
    # in fractions of a beat
    on_beat_notes = [n for n in [1, 2, 3, 4] if min_note_length <= n <= max_note_length]
    # off beat notes with the number required to make them on-beat
    off_beat_notes = [(d, n) for d, n in [(1 / 4, 4), (1 / 3, 3), (1 / 2, 2), (2 / 3, 3), (1.5, 2)] if
                      min_note_length <= d <= max_note_length]
    possible_durations = on_beat_notes + [d for d, n in off_beat_notes]
  rhythm = []
  duration_left = duration
  while duration_left > 0:
    if duration_left > measure_duration:
      current_measure = measure_duration
    else:
      current_measure = duration_left
    duration_left -= current_measure
    # min(possible_durations)
    lo = max(math.ceil(velocity_per_measure - velocity_tolerance), min_note_count)
    hi = math.floor(velocity_per_measure + velocity_tolerance)
    possible_rhythms = []
    max_tries = 8
    while not possible_rhythms and max_tries > 0:
      # print('possible_durations:', possible_durations, 'current_measure:', current_measure, 'lo:', lo, 'hi:', hi)
      possible_rhythms = find_combinations(possible_durations, current_measure, lo, hi)
      lo = max(lo - 1, min_note_count)
      hi = min(hi + 1, int(current_measure / min(possible_durations)))
      max_tries -= 1
    if max_tries == 0:
      raise ValueError(
        f'Could not create a rhythm with at least {lo} notes of duration {duration} with notes {possible_durations}.')
    for r in possible_rhythms:
      random.shuffle(r)
    random.shuffle(possible_rhythms)
    syncopation_counts = [count_syncopation(r) for r in possible_rhythms]
    ranked_by_syncopation = sorted(
      range(len(possible_rhythms)),
      key=lambda i: abs(syncopation_counts[i] - syncopation_rate))
    rhythm += possible_rhythms[ranked_by_syncopation[0]]
  return rhythm


def generate_section_pattern(min_length: float = 8) -> List[Tuple[int, int]]:
  # A B A B'
  # return [(0, 0), (1, 0), (0, 0), (1, 1)]

  alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  forms = ['strophic', 'medley', 'binary', 'ternary', 'rondo'] + ['pop'] * 4
  # strophic form (AAA)
  # medley form (ABCD), (AA BB CC DD)
  # binary form (AB), (AA BB)
  # ternary form (ABA)
  # rondo form (AB AC AD AE A), or symmetrical (ABA C ABA)
  # western popular music forms AA BA, AB AB, ABAC, AAB
  form = random.choice(forms)
  if form == 'strophic':
    shorthand = random.choice(['AAA', 'AAAAA'])
  elif form == 'medley':
    num_sections = random.randint(3, 5)
    repeats = random.randint(1, 2)
    shorthand = ''.join([alphabet[i] * repeats for i in range(num_sections)])
  elif form == 'binary':
    repeats = random.randint(1, 2)
    shorthand = ''.join([alphabet[i] * repeats for i in range(2)])
  elif form == 'ternary':
    repeats = random.randint(1, 2)
    shorthand = ''.join([alphabet[i] * repeats for i in [0, 1, 0]])
  elif form == 'rondo':
    episodes = random.randint(1, 4)
    symmetrical = random.random() < 0.5
    episode_pattern = list(range(1, episodes + 1))
    if symmetrical:
      episode_pattern += list(reversed(episode_pattern[:-1]))
    shorthand = ''.join(['A' + alphabet[i] for i in episode_pattern]) + 'A'
  elif form == 'pop':
    shorthand = random.choice(['AABA', 'ABAB', 'ABAC', 'AAB'])
  else:
    raise NotImplementedError('Unknown form: ' + form)
  section_pattern = [(alphabet.index(c), 0) for c in shorthand]
  sections = set([section for section, _ in section_pattern])
  if len(shorthand) < min_length:
    # print(min_length)
    mappings = {
      section: generate_section_pattern(min_length / len(sections)) for section in sections
    }
    offset = 0
    for section, pattern in mappings.items():
      mappings[section] = [(s + offset, p) for s, p in pattern]
      offset += len(set([s for s, p in pattern]))
    section_pattern = sum([mappings[section] for section, _ in section_pattern], [])
  # print('Sectional form:', ''.join([alphabet[s] for s, _ in section_pattern]))
  return section_pattern


def transpose(notes: List[Note], by: int) -> List[Note]:
  return [Note(set([pitch + by for pitch in note.pitches]), note.duration) for note in notes]


def convert_note_group_sequence_to_midi(left_hand: List[Note], right_hand: List[Note], save_path, merge_tracks: bool,
                                        tempo: int = 120):
  right_track = 1
  left_track = 0
  channel = 0
  # tempo = tempo
  volume = 100
  midi_file = MIDIFile(2)
  midi_file.addTempo(right_track, 0, tempo)

  for hand in [right_hand, left_hand]:
    if hand == right_hand:
      track, octave, volume = right_track, 4, 100
    else:
      track, octave, volume = left_track, 3, 75
    current_time = 0
    for note in hand:
      for pitch in note.pitches:
        midi_file.addNote(track, channel, pitch + 12 * octave, current_time, note.duration, volume)
      current_time += note.duration

  with open(save_path, 'wb') as output_file:
    midi_file.writeFile(output_file)
  if merge_tracks:
    merge_midi_tracks(save_path, save_path)


def merge_midi_tracks(original_file: str, save_to: str):
  original = MidiFile(original_file)
  # merge the two tracks with the most notes; we assume these are the most important
  selected_tracks = sorted(original.tracks, key=lambda t: len(t), reverse=True)[:2]
  track = merge_tracks(selected_tracks)
  for message in list(track):
    if message.type not in ['note_on', 'note_off', 'set_tempo']:
      track.remove(message)
  original.tracks = [track]
  original.save(save_to)
