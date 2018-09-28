

```{python}
import midiutil
from midiutil import MIDIFile

```


```{bash}
mkdir sounds
```


```{python}
def generate_chord(notes, name):
  track    = 0
  channel  = 0
  time     = 0    # In beats
  duration = 1    # In beats
  tempo    = 60   # In BPM
  volume   = 100  # 0-127, as per the MIDI standard

  MyMIDI = MIDIFile(1)
  MyMIDI.addTempo(track, time, tempo)

  for i, pitch in enumerate(notes):
      MyMIDI.addNote(track, channel, pitch, 0, duration, volume)

  with open("sounds/" + name + ".mid", "wb") as output_file:
      MyMIDI.writeFile(output_file)
```


```{python}
start_names_and_notes = [("A", 69), 
                         ("Bb", 70),
                         ("B", 71),
                         ("C", 72),
                         ("Db", 73),
                         ("D", 74),
                         ("Eb", 75),
                         ("E", 76),
                         ("F", 77),
                         ("Gb", 78),
                         ("G", 79),
                         ("Ab", 80)]

for (name, start_note) in start_names_and_notes:
  generate_chord([start_note, start_note + 4, start_note + 7], name + "_maj")
  generate_chord([start_note, start_note + 3, start_note + 7], name + "_min")
```


```{bash}
mkdir -p ~/packages/share/timidity
touch ~/packages/share/timidity/timidity.cfg
```


```{bash}
ls sounds
touch sounds
```

```{bash}
$HOME/packages/bin/timidity major-scale.mid -Ow 
```
<div class="bash">
    <div class="stdout">
        <pre>Playing major-scale.mid
MIDI file: major-scale.mid
Format: 1  Tracks: 2  Divisions: 960
No instrument mapped to tone bank 0, program 0 - this instrument will not be heard
Output major-scale.wav
No pre-resampling cache hit
Last 1 MIDI events are ignored
Playing time: ~10 seconds
Notes cut: 0
Notes lost totally: 0
</pre>
    </div>
</div>