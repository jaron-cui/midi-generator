from dataclasses import dataclass
from typing import List, Tuple, Set, Any, Union, TypeVar, Dict, Callable


def rotate_left(items: List[Any], n: int) -> List[Any]:
  return items[n % len(items):] + items[:n % len(items)]


_accidentals = {'♮': 0, 'b': -1, '#': 1}


def parse_note(index: Union[int, str]) -> Tuple[int, int]:
  # TODO: this does not currently respect octave
  if isinstance(index, str):
    offset = _accidentals[index[-1]]
    index = int(index[:-1])
  else:
    offset = 0
  return index, offset


class Key:
  """
  Describes a musical key.
  """
  def __init__(self, pitches: List[int]):
    self._pitches = pitches

  @classmethod
  def major(cls):
    return Key([0, 2, 4, 5, 7, 9, 11])

  @classmethod
  def minor(cls):
    return Key([0, 2, 3, 5, 7, 8, 10])

  @classmethod
  def diminished(cls):
    return Key([0, 2, 3, 5, 6, 8, 9, 11])

  @classmethod
  def augmented(cls):
    return Key([0, 3, 4, 7, 8, 11])

  @property
  def pitches(self) -> List[int]:
    return self._pitches.copy()

  def __getitem__(self, index: int | str) -> 'Key':
    # TODO: this does not currently respect octave
    index, offset = parse_note(index)
    index -= 1
    return Key([pitch + offset for pitch in rotate_left(self.pitches, index)])

  def pitch(self, index: int | str) -> int:
    if isinstance(index, str):
      offset = _accidentals[index[-1]]
      index = int(index[:-1])
    else:
      offset = 0
    index -= 1
    relative_index = index % len(self.pitches)
    octave = index // len(self.pitches)
    return self.pitches[relative_index] + octave * 12 + offset

  def chord(self, notes: List[int], required: List[int] = None, cardinality: int = None, inversion: int = 1,
            exclusion_preference: List[int] = None):
    if required is None:
      required = notes
    if cardinality is None:
      cardinality = len(notes)
    if exclusion_preference is None:
      exclusion_preference = []

    included_notes = set(required)
    remaining_notes = set(notes) - included_notes
    remaining_preferred_notes = remaining_notes - set(exclusion_preference)
    remaining_un_preferred_notes = remaining_notes - remaining_preferred_notes

    while len(included_notes) < cardinality and (remaining_preferred_notes or remaining_un_preferred_notes):
      if remaining_preferred_notes:
        included_notes.add(remaining_preferred_notes.pop())
      else:
        included_notes.add(remaining_un_preferred_notes.pop())
    # TODO: this does not properly do inversions respecting octave
    return rotate_left([self.pitch(note) for note in notes], inversion - 1)

  def as_root(self):
    return self[-min(self.pitches)]

  def is_major_like(self):
    # check that note 3 is a major third from the base note
    return self.as_root().pitch(3) == 4

  def is_minor_like(self):
    # check that note 3 is flatted
    return self.as_root().pitch(3) == 3

  def is_diminished_like(self):
    # check that notes 3 and 5 are flatted
    as_root = self.as_root()
    return as_root.pitch(3) == 3 and as_root.pitch(5) == 6

  def is_augmented_like(self):
    # check that note 5 is sharped
    return self.as_root().pitch(5) == 8

  def pitch_as_scale_note(self, pitch: int) -> Union[int, str]:
    if pitch in self.pitches:
      return self.pitches.index(pitch)
    if pitch - 1 in self.pitches:
      return f'{self.pitches.index(pitch - 1)}#'
    if pitch + 1 in self.pitches:
      return f'{self.pitches.index(pitch + 1)}b'
    raise ValueError(f'Unable to identify pitch {pitch} in relation to a scale note of {self}.')

  def __str__(self) -> str:
    return ' '.join([str(pitch) for pitch in self.pitches])


@dataclass
class Note:
  """
  Represents a generated collection of notes played simultaneously.
  """
  pitches: Set[int]
  duration: float


@dataclass
class Chord:
  """
  Represents a chord in a chord progression.
  Describes the key of the chord, the notes of the key which are chord notes,
  and additional parameters for the chord quality.
  """
  key: Key
  notes: List[int]
  duration: float
  required: List[int] = None
  cardinality: int = None
  inversion: int = 1
  exclusion_preference: List[int] = None

  def copy(self) -> 'Chord':
    return Chord(self.key, self.notes, self.duration, self.required, self.cardinality, self.inversion,
                 self.exclusion_preference)

  @staticmethod
  def standard(key: Key, duration: float) -> 'Chord':
    return Chord(key, [1, 3, 5], duration)

  @staticmethod
  def seventh(key: Key, duration: float) -> 'Chord':
    # TODO: flat7
    return Chord(key, [1, 3, 5, 7], duration)

  @staticmethod
  def sus2(key: Key, duration: float) -> 'Chord':
    return Chord(key, [1, 2, 5], duration)


@dataclass
class CompatibleElements:
  blocks: List['CompatibleElements'] = None
  block_type: str = None
  start_pitch: int = None
  end_pitch: int = None
  spec: Dict = None


@dataclass
class TrackCharacteristics:
  # beats in a measure
  measure_length: int
  # approximate number of notes per measure
  melody_velocity_bounds: Tuple[float, float]
  # how many measures it should take to ramp from min to max velocity
  velocity_inertia: float
  # approximate number of syncopations per measure
  melody_syncopation_bounds: Tuple[float, float]
  # average numbers of notes harmonized
  harmonization_bounds: Tuple[float, float]
  beat_harmonization_rate_bounds: Tuple[float, float]
  # how likely a block is to try and borrow from other blocks
  borrow_likelihood_bounds: Tuple[float, float]
  # length of phrases
  phrase_length_bounds: Tuple[int, int]
  max_pitch_interval: int
  home_octave: int
  octave_bounds: Tuple[int, int]
  substructures_per_block_bounds: Tuple[float, float]
  substructure_size_bounds: Tuple[float, float]
  note_duration_bounds: Tuple[float, float]
  phrase_block_rate: float
  phrase_block_syncopation_rate: float


@dataclass
class SectionCharacteristics:
  track_characteristics: TrackCharacteristics
  measure_length: int
  velocity: float
  velocity_tolerance: float
  syncopation: float
  melody_syncopation_bounds: Tuple[float, float]
  harmonization_bounds: Tuple[float, float]
  beat_harmonization_rate: float
  borrow_likelihood: float
  max_pitch_interval: int
  pitch_bounds: Tuple[int, int]
  substructures_per_block: float
  substructure_size_bounds: Tuple[float, float]
  note_duration_bounds: Tuple[float, float]

  def copy(self) -> 'SectionCharacteristics':
    return SectionCharacteristics(self.track_characteristics, self.measure_length, self.velocity,
                                  self.velocity_tolerance, self.syncopation, self.melody_syncopation_bounds,
                                  self.harmonization_bounds, self.beat_harmonization_rate, self.borrow_likelihood,
                                  self.max_pitch_interval, self.pitch_bounds, self.substructures_per_block,
                                  self.substructure_size_bounds, self.note_duration_bounds)

  def evolve(self, timestep: float) -> 'SectionCharacteristics':
    # TODO
    copy = self.copy()
    template = copy.track_characteristics
    return copy


# @dataclass
# class Phrase(ABC):
#   type: str
#   start_pitch: int
#   end_pitch: int
#   chord_progression: List[Chord]
#
# @dataclass
# class SequencePhrase(Phrase):
#   type: 'sequence'


@dataclass
class ChordCharacter:
  tone_change_likelihood: float
  augment_likelihood: float
  diminish_likelihood: float
  suspend_likelihood: float
