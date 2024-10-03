import threading

import mido
from mido import MidiFile
import time
from pygame import mixer


freqs = {
  'C': 261.63,
  'C#': 277.18,
  'D': 293.66,
  'D#': 311.13,
  'E': 329.63,
  'F': 349.23,
  'F#': 369.99,
  'G': 392.00,
  'G#': 413.30,
  'A': 440.00,
  'A#': 466.16,
  'B': 493.88,
}
mixer.init()
mixer.music.load('../abreezefromalabama.mid')
mixer.music.play()
while mixer.music.get_busy():
  time.sleep(1)
# mixer.set_num_channels(80)

# notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# sounds = {note: mixer.Sound(f'../notes/{note}.mp3') for note in notes}
#
#
# def play_note(note):
#   sounds[note].play(maxtime=800)
#
# time.sleep(1)
# for note in notes:
#   play_note(note)
#   time.sleep(0.1)
# time.sleep(1)
#
# def note_from_code(code: int) -> str:
#
#   return notes[code % 12]
#
#
# mid = MidiFile('../abreezefromalabama.mid')
# # player = musicalbeeps.Player(volume=0.3, mute_output=False)
# def play_track(track):
#   for msg in track:
#     if msg.type == 'note_on':
#       note = note_from_code(msg.note)
#       play_note(note)
#       if msg.time != 0:
#         time.sleep(msg.time / 500)
#     elif msg.type == 'note_off':
#       if msg.time != 0:
#         time.sleep(msg.time / 500)
#     else:
#       print(msg)
#     print(msg)
# for i, track in enumerate(mid.tracks):
#   thread = threading.Thread(target=play_track, args=(track,))
#   thread.start()
#
