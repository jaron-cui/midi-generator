from musiclang import Score
from musiclang.library import *

score = Score.from_midi('../output/midi_webscrape/bwv772.mid')
chords = score.to_chords_with_duration()
print(chords)
notes = r
for chord in chords:
  # chord.ton
  # arpeggio = chord.chord_notes[0].e
  # for note in chord.chord_notes[1:]:
  #   arpeggio += note.e
  # notes += arpeggio
  notes += chord.to_voicing()

notes.to_midi('test.mid')
