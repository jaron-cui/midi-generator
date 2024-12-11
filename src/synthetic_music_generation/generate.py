import time

import numpy
from overrides import override
import math
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Any, Union, TypeVar, Dict, Callable
from dataclasses import dataclass
from itertools import combinations

from synthetic_music_generation.representations import Note, Chord, CompatibleElements, SectionCharacteristics, Key, \
  TrackCharacteristics, ChordCharacter
from synthetic_music_generation.util import generate_arpeggio_possibilities, create_notes, generate_run_possibilities, \
  calculate_interval, chord_excerpt, next_scale_pitch, next_chord_pitch, item_at_time, generate_rhythm, \
  generate_section_pattern, transpose, convert_note_group_sequence_to_midi


# def adapt_accidentals(notes: List[Note], src_chord_progression: List[Chord], tgt_chord_progression: List[Chord]) -> List[Note]:
#   new_notes = []
#   current_time = 0
#   for note in notes:
#     src_chord = item_at_time(src_chord_progression, current_time)
#     tgt_chord = item_at_time(tgt_chord_progression, current_time)
#     relative_difference = src_chord.key.pitch_as_scale_note(tgt_chord.key.pitch(0))
#     scale_indices = {src_chord.key.pitch_as_scale_note(pitch) for pitch in note.pitches}
#     adapted_pitches = {tgt_chord.key.pitch(index) for index in scale_indices}
#     new_notes.append(Note(adapted_pitches, note.duration))
#   return new_notes



class NoteBlock(ABC):
  def __init__(self, block_type: str, start_pitch: int, end_pitch: int, chord_progression: List[Chord],
               obscured: bool = False):
    self.block_type = block_type
    self.start_pitch = start_pitch
    self.end_pitch = end_pitch
    self.chord_progression = chord_progression
    self.obscured = obscured
    self.root_key = Key.major()
    self._notes = None
    self.spec = {}

  @property
  def duration(self) -> float:
    return sum(c.duration for c in self.chord_progression)

  def find_duration_matches(self, duration: float) -> List['NoteBlock']:
    if sum([chord.duration for chord in self.chord_progression]) == duration:
      return [self]

  def find_harmonic_matches(self, block: 'Noteblock') -> List['NoteBlock']:
    """
    Extracts note blocks that fit the harmonic and duration requirements provided.
    These can be found exactly, or be derived from blocks which match in duration and when transposed
    remain compatible with the required chord progression.
    :param block:
    :return:
    """
    if self.start_pitch == block.start_pitch and self.end_pitch == block.end_pitch:
      if self.chord_progression == block.chord_progression:
        return [self]
    return []

  @abstractmethod
  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    pass

  @abstractmethod
  def generate_notes(self, motif_bank: 'NoteBlock', section_specs: SectionCharacteristics):
    pass

  def get_notes(self) -> List[Note]:
    if self._notes is None:
      raise RuntimeError('Notes have not been generated yet. Call the generate function.')
    return self._notes


class SequentialNoteBlock(NoteBlock, ABC):
  def __init__(self, block_type: str, start_pitch: int, end_pitch: int, chord_progression: List[Chord]):
    super().__init__(block_type, start_pitch, end_pitch, chord_progression)
    self.subtype = None
    self.blocks: Union[List[NoteBlock] | None] = None

  def generate_notes(self, motif_bank: NoteBlock, section_specs: SectionCharacteristics):
    self.blocks = self._generate_blocks(motif_bank, section_specs)
    for block in self.blocks:
      block.generate_notes(motif_bank, section_specs)
      motif_bank = ListNoteBlock(motif_bank, block)

  @abstractmethod
  def _generate_blocks(self, motif_bank: NoteBlock, section_specs: SectionCharacteristics) -> List[NoteBlock]:
    pass

  @override
  def get_notes(self) -> List[Note]:
    return sum([block.get_notes() for block in self.blocks], [])

  def find_duration_matches(self, duration: float) -> List['NoteBlock']:
    matches = []
    if sum([chord.duration for chord in self.chord_progression]) == duration:
      matches.append(self)
    for block in self.blocks:
      matches.extend(block.find_duration_matches(duration))
    return matches

  def find_harmonic_matches(self, start_pitch: int, end_pitch: int, chord_progression: List[Chord]) -> List[
    'NoteBlock']:
    matches = []
    if self.start_pitch == start_pitch and self.end_pitch == end_pitch:
      # TODO: check broader harmonic match, not just exact chord match
      if self.chord_progression == chord_progression:
        matches.append(self)
    for block in self.blocks:
      # TODO: check subsequences, not just each sub-block
      matches.extend(block.find_harmonic_matches(start_pitch, end_pitch, chord_progression))
    return matches


class ListNoteBlock(SequentialNoteBlock):
  def __init__(self, *note_blocks: NoteBlock):
    super().__init__('list', note_blocks[0].start_pitch, note_blocks[-1].end_pitch,
                     sum([n.chord_progression for n in note_blocks], []))
    self.blocks = list(note_blocks)

  def _generate_blocks(self, motif_bank: NoteBlock, section_specs: SectionCharacteristics) -> List[NoteBlock]:
    return self.blocks

  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    compatible_elements = []
    if block.block_type == 'list':
      # find subsequence block matches for block
      for i in range(len(self.blocks) - len(block.blocks)):
        subsequence = self.blocks[i:i + len(block.blocks)]
        sub_duration = sum([b.duration for b in subsequence])
        if sub_duration > block.duration:
          break
        # check that each corresponding block has the same relative pitch progression
        matches = []
        p1, p2 = None, None
        for (b1, b2) in zip(subsequence, block.blocks):
          if b1.duration != b2.duration or b1.block_type != b2.block_type:
            break
          # matching block types and sizes
          existing_interval = calculate_interval(
            b1.root_key.pitch_as_scale_note(b1.start_pitch),
            b1.root_key.pitch_as_scale_note(b1.end_pitch))
          proposed_interval = calculate_interval(
            b2.root_key.pitch_as_scale_note(b2.start_pitch),
            b2.root_key.pitch_as_scale_note(b2.end_pitch))
          if existing_interval != proposed_interval:
            break
          if p1 is not None:
            existing_interval = calculate_interval(
              b1.root_key.pitch_as_scale_note(p1.end_pitch),
              b1.root_key.pitch_as_scale_note(b1.start_pitch))
            proposed_interval = calculate_interval(
              b2.root_key.pitch_as_scale_note(p2.end_pitch),
              b2.root_key.pitch_as_scale_note(b2.start_pitch))
            if existing_interval != proposed_interval:
              break
          p1, p2 = b1, b2
          # record the details of the compatible block and include the specification
          # from which aspects can be derived instead of generated anew
          matches.append(CompatibleElements(
            block_type=b1.block_type,
            start_pitch=b1.start_pitch,
            end_pitch=b1.end_pitch,
            spec=b1.spec
          ))
        if len(matches) == len(subsequence):
          subsequence_match = CompatibleElements(
            block_type='list',
            start_pitch=subsequence[0].start_pitch,
            end_pitch=subsequence[-1].end_pitch
          )
          compatible_elements.append(subsequence_match)
    for sub_block in self.blocks:
      if sub_block.duration >= block:
        compatible_elements.extend(sub_block.find_compatible_elements(block))
    return compatible_elements


class HarmonyBlock(NoteBlock):
  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    # TODO: implement matching
    return []

  def generate_notes(self, motif_bank: 'NoteBlock', section_specs: SectionCharacteristics):
    # if random.random() < section_specs.beat_harmonization_rate:
    #   # harmonize
    #
    self._notes = [Note({self.start_pitch}, self.duration)]

  def __init__(self, pitch: int, note: Note, chord: Chord):
    super().__init__('note', pitch, pitch, chord_excerpt([chord], 0, note.duration))
    self.chord = chord
    self.note = note

  def find_duration_matches(self, duration: float) -> List['NoteBlock']:
    if self.duration == duration:
      return [self]


class RhythmBlock(NoteBlock):
  def __init__(self, pitch: int, chord_progression: List[Chord]):
    super().__init__('rhythm', pitch, pitch, chord_progression)

  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    return []

  def generate_notes(self, motif_bank: 'NoteBlock', section_specs: SectionCharacteristics):
    min_d, max_d = section_specs.note_duration_bounds
    rhythm = generate_rhythm(
      self.duration,
      min_d,
      max_d,
      section_specs.syncopation,
      section_specs.velocity,
      velocity_tolerance=section_specs.velocity_tolerance * 2,
      measure_duration=self.duration)
    self._notes = create_notes([self.start_pitch] * len(rhythm), rhythm, self.chord_progression, section_specs)
    # self._notes = [Note({self.start_pitch}, duration) for duration in rhythm]


class TrillBlock(NoteBlock):
  def __init__(self, chord_progression: List[Chord], start_pitch: int, end_pitch: int = None):
    if end_pitch is None:
      end_pitch = start_pitch
      # trill can be with an adjacent note or chord notes
      trill_type = random.choice(['adjacent', 'chord'])
      super().__init__('trill', start_pitch, end_pitch, chord_progression)
      self.spec['trill_type'] = trill_type
    else:
      super().__init__('trill', start_pitch, end_pitch, chord_progression)
      self.spec['trill_type'] = 'predefined'

  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    return []

  def generate_notes(self, motif_bank: 'NoteBlock', section_specs: SectionCharacteristics):
    min_note_duration, max_note_duration = section_specs.note_duration_bounds
    rhythm = generate_rhythm(
      self.duration,
      min_note_duration,
      max_note_duration,
      section_specs.syncopation,
      section_specs.velocity * 2,
      velocity_tolerance=1)
    # TODO: make this selection better and also add support for ternary trills
    cycle_type = random.choice([[(0, 1), (1, 0)], [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]])
    cycle = random.choice(cycle_type)
    self.spec['rhythm'] = rhythm
    self.spec['cycle'] = cycle

    trill_type, rhythm, cycle = self.spec['trill_type'], self.spec['rhythm'], self.spec['cycle']
    num_unique_nodes = len(set(cycle))
    nodes = [lambda d, __: Note({self.start_pitch}, d)]
    start_pitch = self.start_pitch
    for i in range(1, num_unique_nodes):
      select_node = None
      if trill_type == 'predefined':
        select_node = lambda d, __: Note({self.end_pitch}, d)
      elif trill_type == 'adjacent':
        direction = random.choice([-1, 1])
        select_node = lambda d, chord: Note({next_scale_pitch(start_pitch, chord, direction)}, d)
      elif trill_type == 'chord':
        direction = random.choice([-1, 1])
        # TODO: this should depend on a section-wide verbosity parameter
        cardinality = random.choice([1, 2, 3])

        def get_next_chord_pitches(pitch: int, chord: Chord) -> Set[int]:
          pitches = set([])
          chord_pitch = next_chord_pitch(pitch, chord, direction)
          for _ in range(cardinality):
            pitches.add(chord_pitch)
            chord_pitch = next_chord_pitch(pitch, chord, direction)
          return pitches

        select_node = lambda d, chord: Note(get_next_chord_pitches(start_pitch, chord), d)
      nodes.append(select_node)
    self._notes = []
    cycle_index = 0
    current_time = 0
    for duration in rhythm:
      self._notes.append(nodes[cycle_index](duration, item_at_time(self.chord_progression, current_time)))
      cycle_index = (cycle_index + 1) % len(nodes)
      current_time += duration


class StagnateBlock(SequentialNoteBlock):
  """
  Stagnation structures include:
  1. Pitch repetition
  2. Trills
  3. Arpeggios
  4. Holds
  5. Scales
  """

  def __init__(self, pitch: int, chord_progression: List[Chord]):
    super().__init__('stagnate', pitch, pitch, chord_progression)

  def _generate_blocks(self, motif_bank: NoteBlock, section_specs: SectionCharacteristics) -> List[NoteBlock]:
    min_substructure_size, max_substructure_size = section_specs.substructure_size_bounds
    min_substructure_size = min(max((int(min_substructure_size * 2) / 2), 0.5), self.duration)
    # print('min_substructure_size:', min_substructure_size)
    subdivisions = generate_rhythm(
      self.duration,
      min_substructure_size,
      max_substructure_size,
      section_specs.syncopation,
      section_specs.substructures_per_block * self.duration / section_specs.measure_length,
      measure_duration=self.duration,
      possible_durations=list(
        numpy.arange(min_substructure_size, max_substructure_size, section_specs.note_duration_bounds[0])))
    block_types = {
      'rhythm': 0,
      'trill': 1,
      'arpeggio': 3 * section_specs.note_duration_bounds[0],
      'run': 3 * section_specs.note_duration_bounds[0]
    }
    self.blocks: List[NoteBlock] = []
    current_time = 0
    for subdivision in subdivisions:
      legal_block_types = {block_type: min_duration for block_type, min_duration in block_types.items() if
                           subdivision >= min_duration}
      # TODO: more sophisticated probability selection system
      block_type = random.choice(list(legal_block_types.keys()))
      block_chords = chord_excerpt(self.chord_progression, current_time, current_time + subdivision)
      if block_type == 'rhythm':
        block = RhythmBlock(self.start_pitch, block_chords)
      elif block_type == 'trill':
        block = TrillBlock(block_chords, self.start_pitch)
      elif block_type == 'arpeggio':
        # print(f'arpeg, {subdivision}, min: {legal_block_types[block_type]}')
        block = ArpeggioBlock(self.start_pitch, self.start_pitch, block_chords, random.choice([True, False]))
      elif block_type == 'run':
        # print(legal_block_types)
        block = RunBlock(self.start_pitch, self.start_pitch, block_chords, random.choice([True, False]))
      else:
        raise RuntimeError('Unknown block type:', block_type)
      self.blocks.append(block)
      current_time += subdivision
    # pitch = self.start_pitch
    # chord_progression = self.chord_progression
    # duration = sum([chord.duration for chord in chord_progression])
    # return [NBlock(pitch, Note({pitch}, duration), chord_progression[0])]
    return self.blocks

  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    # TODO: check if block matches subsequences
    compatible_elements = []
    if self.duration < block.duration:
      return compatible_elements
    if self.block_type == block.block_type and self.duration == block.duration:
      existing_interval = calculate_interval(
        self.root_key.pitch_as_scale_note(self.start_pitch),
        self.root_key.pitch_as_scale_note(self.end_pitch))
      proposed_interval = calculate_interval(
        self.root_key.pitch_as_scale_note(block.start_pitch),
        self.root_key.pitch_as_scale_note(block.end_pitch))
      if existing_interval == proposed_interval:
        # TODO: check that chords line up or something
        # including start and end pitches indicate the notes themselves may be transferable
        compatible_elements.append(CompatibleElements(
          blocks=[CompatibleElements(
            block_type=b.block_type,
            start_pitch=b.start_pitch,
            end_pitch=b.end_pitch,
            spec=b.spec
          ) for b in self.blocks],
          block_type=self.block_type,
          start_pitch=self.start_pitch,
          end_pitch=self.end_pitch,
          spec=self.spec
        ))
      else:
        # if nothing else, at least the rhythm
        compatible_elements.append(CompatibleElements(
          blocks=[CompatibleElements(
            block_type=b.block_type,
            spec=b.spec
          ) for b in self.blocks],
          block_type=self.block_type,
          spec=self.spec
        ))
    for sub_block in self.blocks:
      compatible_elements.extend(sub_block.get_compatible_elements(block))
    return compatible_elements


class TransitionBlock(SequentialNoteBlock):
  def __init__(self, start_pitch: int, end_pitch: int, chord_progression: List[Chord]):
    super().__init__('transition', start_pitch, end_pitch, chord_progression)

  def _generate_blocks(self, motif_bank: NoteBlock, section_specs: SectionCharacteristics) -> List[NoteBlock]:
    min_substructure_size, max_substructure_size = section_specs.substructure_size_bounds
    min_substructure_size = min(max((int(min_substructure_size * 2) / 2), 0.5), self.duration)
    subdivisions = generate_rhythm(
      self.duration,
      min_substructure_size,
      max_substructure_size,
      section_specs.syncopation,
      section_specs.substructures_per_block * self.duration / section_specs.measure_length,
      measure_duration=self.duration,
      possible_durations=list(
        numpy.arange(min_substructure_size, max_substructure_size, section_specs.note_duration_bounds[0])))
    block_types = {
      'arpeggio': 3 * section_specs.note_duration_bounds[0],
      'run': 3 * section_specs.note_duration_bounds[0]
    }
    self.blocks: List[NoteBlock] = []
    current_time = 0
    for subdivision in subdivisions:
      block_chords = chord_excerpt(self.chord_progression, current_time, current_time + subdivision)
      legal_block_types = {block_type: min_duration for block_type, min_duration in block_types.items() if
                           subdivision >= min_duration}
      block = None
      # TODO: replace rhythm with pickup/pickdown? something for short default
      if legal_block_types:
        # TODO: more sophisticated probability selection system
        block_type = random.choice(list(legal_block_types.keys()))
        if block is None:
          pass
        if block_type == 'arpeggio':
          block = ArpeggioBlock(self.start_pitch, self.end_pitch, block_chords, random.choice([True, False]))
        elif block_type == 'run':
          block = RunBlock(self.start_pitch, self.end_pitch, block_chords, random.choice([True, False]))
        else:
          raise RuntimeError('Unknown block type:', block_type)
      else:
        block = RhythmBlock(self.start_pitch, block_chords)
      self.blocks.append(block)
      current_time += subdivision
    # pitch = self.start_pitch
    # chord_progression = self.chord_progression
    # duration = sum([chord.duration for chord in chord_progression])
    # return [NBlock(pitch, Note({pitch}, duration), chord_progression[0])]
    return self.blocks
    # TODO: cleanup
    # duration = self.duration
    # if duration > 0.5:
    #   notes = [Note({self.start_pitch}, 0.5), Note({self.end_pitch}, duration - 0.5)]
    # else:
    #   notes = [Note({self.start_pitch}, duration)]
    # return [NBlock(list(note.pitches)[0], note, self.chord_progression[0]) for note in notes]

  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    # TODO: implement matching
    return []


class PickupBlock(TransitionBlock):
  pass


# make an arpeggio block generating function that tries a bunch of options
# then evaluates the arpeggios for:
# 1. consistency of intervals
#


class ArpeggioBlock(NoteBlock):
  def __init__(self, start_pitch: int, end_pitch: int, chord_progression: List[Chord], include_end_pitch: bool):
    super().__init__('arpeggio', start_pitch, end_pitch, chord_progression)
    self.spec['include_end_pitch'] = include_end_pitch

  def generate_notes(self, motif_bank: 'NoteBlock', section_specs: SectionCharacteristics):
    # TODO: look in the motif bank. if an applicable rhythm is found, use that
    #       if an applicable number of peaks is found, use that. intervals, peak values, etc...
    #       just take an entire pre-generated arpeggio if it works
    #       maybe implement some 'nudge-arpeggio' or 'adjust-arpeggio' that redistributes some rhythms
    # if self.duration == 1.0:
    #   print('WELL WELL WEL')
    possible_arpeggiations = generate_arpeggio_possibilities(
      self.start_pitch, self.end_pitch, self.chord_progression, self.spec['include_end_pitch'], section_specs)
    selected_arpeggios = random.choice(possible_arpeggiations)
    self.spec['arpeggios'] = selected_arpeggios
    pitches, rhythm = [], []
    for section_pitches, section_rhythm in selected_arpeggios:
      pitches.extend(section_pitches)
      rhythm.extend(section_rhythm)
    self._notes = create_notes(pitches, rhythm, self.chord_progression, section_specs)

  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    # TODO: implement matching
    return []


class RunBlock(NoteBlock):
  def __init__(self, start_pitch: int, end_pitch: int, chord_progression: List[Chord], include_last_pitch: bool):
    super().__init__('run', start_pitch, end_pitch, chord_progression)
    self.spec['include_last_pitch'] = include_last_pitch

  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    # TODO: implement matching
    return []

  def generate_notes(self, motif_bank: 'NoteBlock', section_specs: SectionCharacteristics):
    possible_runs = generate_run_possibilities(self.start_pitch, self.end_pitch, self.chord_progression,
                                               self.spec['include_last_pitch'], section_specs)
    selected_runs = random.choice(possible_runs)
    self.spec['runs'] = selected_runs
    pitches, rhythm = [], []
    for section_pitches, section_rhythm in selected_runs:
      pitches.extend(section_pitches)
      rhythm.extend(section_rhythm)
    self._notes = create_notes(pitches, rhythm, self.chord_progression, section_specs)


class EmptyNoteBlock(NoteBlock):
  def __init__(self):
    super().__init__('empty', 0, 0, [])

  def find_compatible_elements(self, block: 'NoteBlock') -> List[CompatibleElements]:
    return []

  def generate_notes(self, motif_bank: 'NoteBlock', section_specs: SectionCharacteristics):
    self._notes = []


def generate_spec() -> SectionCharacteristics:
  measure_length = random.choice([3, 4])
  velocity = random.uniform(0.5, measure_length / 0.5)
  velocity_tolerance = random.uniform(1, 4)
  syncopation = random.uniform(0, 2.4)
  max_harmonization = random.randint(1, 4)
  beat_harmonization_rate = random.uniform(0.2, 1.1)
  substructure_size_bounds = random.uniform(0.5, 4), 8
  phrase_block_rate = random.uniform(0.5, 2)
  phrase_length_bounds = random.randint(1, 2), random.randint(2, 8)
  max_substructures_per_block = random.randint(1, 6)
  template = TrackCharacteristics(
    measure_length=measure_length,
    melody_velocity_bounds=(3, 8),
    velocity_inertia=2,
    melody_syncopation_bounds=(0, 2),
    harmonization_bounds=(0, max_harmonization),
    beat_harmonization_rate_bounds=(0, 0.5),
    borrow_likelihood_bounds=(0.1, 0.8),
    phrase_length_bounds=phrase_length_bounds,
    max_pitch_interval=14,
    home_octave=0,
    octave_bounds=(0, 3),
    substructure_size_bounds=substructure_size_bounds,
    substructures_per_block_bounds=(0.5, max_substructures_per_block),
    note_duration_bounds=(0.5, 4),
    phrase_block_rate=phrase_block_rate,
    phrase_block_syncopation_rate=0.2
  )
  section_specs = SectionCharacteristics(
    track_characteristics=template,
    measure_length=measure_length,
    velocity=velocity,
    velocity_tolerance=velocity_tolerance,
    syncopation=syncopation,
    melody_syncopation_bounds=(0.0, 1.0),
    harmonization_bounds=(0, max_harmonization),
    beat_harmonization_rate=beat_harmonization_rate,
    borrow_likelihood=0.3,
    max_pitch_interval=14,
    pitch_bounds=(0, 1000),
    substructures_per_block=random.uniform(*template.substructures_per_block_bounds),
    substructure_size_bounds=substructure_size_bounds,
    note_duration_bounds=(0.5, 4)
  )
  return section_specs


def generate_chord_progression(duration: float, character: ChordCharacter, measure_length: int) -> List[Chord]:
  duration_splits = {
    3: [1.5, 3],
    4: [2, 4]
  }
  rhythm = generate_rhythm(duration, 0, 10, 0, 1.2, 0.7, measure_duration=measure_length,
                           possible_durations=duration_splits.get(measure_length, [measure_length]))
  key: Key = random.choice([Key.major(), Key.minor()])
  chord_progression = []
  for duration in rhythm:
    relative_key = key[random.randint(1, len(key.pitches))]
    # TODO: implement tone changes, making keys diminished, etc...
    # if random.random() < character.tone_change_likelihood:
    #   relative_key = random.choice([relative_key., relative_key.minor])
    chord_progression.append(Chord.standard(relative_key, duration))
  return chord_progression


def generate_pitch_changes(rhythm: List[float], chord_progression: List[Chord]) -> List[Tuple[int, int]]:
  # TODO: this kinda sucks; replace it entirely
  directions = [-1, 1]
  direction = random.choice(directions)
  roll_direction_probability = 0.7
  chord_note_probability = 0.2
  pitch_changes = []
  start_note = random.choice(chord_progression[0].key.pitches)
  current_time = 0
  for duration in rhythm:
    # if random.random() < chord_note_probability:
    #   end_note = next_chord_pitch(start_note, item_at_time(chord_progression, current_time), direction)
    # else:
    #   end_note = next_scale_pitch(start_note, item_at_time(chord_progression, current_time), direction)
    chord = item_at_time(chord_progression, current_time)
    end_note = random.choice(chord.key.pitches)
    pitch_changes.append((start_note, end_note))
    start_note = end_note
    current_time += duration
    if random.random() < roll_direction_probability:
      direction = random.choice(directions)
  return pitch_changes


def generate_section() -> List[Note]:
  section_specs = generate_spec()
  template = section_specs.track_characteristics
  phrase_pattern = generate_section_pattern()
  num_phrases = len(set(p[0] for p in phrase_pattern))
  min_phrase_length, max_phrase_length = template.phrase_length_bounds
  m = template.measure_length
  phrases = []
  for phrase_index in range(num_phrases):
    num_measures = random.randint(min_phrase_length, max_phrase_length)
    # we scale measures down to the length of a beat so that we can control block syncopation rate without
    # modifying the generate_rhythm function's syncopation detection
    # print('BLOCK RHY<:', [(n + 1) / m for n in range(2 * m)])
    block_rhythm = generate_rhythm(float(num_measures), 0, 10, template.phrase_block_syncopation_rate,
                                   template.phrase_block_rate * num_measures, measure_duration=num_measures,
                                   possible_durations=[(n + 1) / m for n in range(2 * m)])
    block_rhythm = [round(n * m) for n in block_rhythm]
    chord_progression = generate_chord_progression(num_measures * m, None, m)
    pitch_changes = generate_pitch_changes(block_rhythm, chord_progression)
    blocks = []
    current_time = 0
    for block_duration, (start_pitch, end_pitch) in zip(block_rhythm, pitch_changes):
      block_type = random.choice(['stagnate', 'transition'])
      block_chords = chord_excerpt(chord_progression, current_time, current_time + block_duration)
      if block_type == 'stagnate':
        blocks.append(StagnateBlock(start_pitch, block_chords))
      elif block_type == 'transition':
        blocks.append(TransitionBlock(start_pitch, end_pitch, block_chords))
    phrases.append(ListNoteBlock(*blocks))
  section_phrases = [phrases[phrase_index] for (phrase_index, phrase_variant) in phrase_pattern]
  note_block = ListNoteBlock(*section_phrases)
  for phrase in phrases:
    phrase.generate_notes(EmptyNoteBlock(), section_specs)
  return note_block.get_notes()


def generate_piece(save: str):
  tries = 100
  while tries > 0:
    try:
      section = generate_section()
      transpose_by = random.randint(-12, 12)
      section = transpose(section, transpose_by)
      break
    except (RuntimeError, ValueError):
      tries -= 1
  else:
    raise RuntimeError('Could not generate piece.')
  tempo = random.randint(40, 200)
  convert_note_group_sequence_to_midi([], section, save, tempo=tempo)


if __name__ == '__main__':
  generate_piece('temp.mid')
  from pygame import mixer
  mixer.init()
  mixer.set_num_channels(80)
  mixer.music.load('temp.mid')
  mixer.music.play()
  while mixer.music.get_busy():
    time.sleep(1)




