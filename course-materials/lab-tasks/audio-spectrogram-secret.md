# Lab task: hide a picture in a sound

Session of 13 November 2026 · individual task · about 45 minutes

## Idea

A spectrogram shows which frequencies are in a sound over time. If you build
a sound whose frequencies follow the pixels of a picture, the picture shows
up in the spectrogram. You cannot see it in the waveform and you can hardly
hear it. The Fourier transform finds it.

You need only numpy and opencv. Run everything from the `src/` folder.
Results land in `output/`.

## Part 1: try it (10 min)

1. Hide the dog in a sound:

   ```
   python spectrogram_secret.py hide --image ../data/dog.jpg --out dog.wav
   ```

2. Play `output/dog.wav`. What do you hear? Could you guess the picture?

3. Look at its spectrogram:

   ```
   python spectrogram_secret.py show --wav ../output/dog.wav
   ```

   Open the PNG in `output/`.

## Part 2: your own secret (15 min)

Hide something of your own: a photo, a drawing, a logo, or a text.

```
python spectrogram_secret.py hide --text "HELLO" --out secret.wav
python spectrogram_secret.py hide --image my_picture.jpg --out secret.wav
```

Pictures with strong contrast and simple shapes work best. Try at least two
and keep the better one.

Then swap WAV files with your neighbour. Do not say what is in it. Decode
each other's secret with `show`.

## Part 3: the window (15 min)

The spectrogram cuts the sound into short windows and runs an FFT on each
one. The window size is a trade-off. Look at the same sound three times:

```
python spectrogram_secret.py show --wav ../output/secret.wav --window 256
python spectrogram_secret.py show --wav ../output/secret.wav --window 1024
python spectrogram_secret.py show --wav ../output/secret.wav --window 4096
```

The script prints the frequency resolution and the time resolution each
time. Compare the three pictures:

- Which edges get sharper with a small window, which with a large one?
- Why can you not have both at once?

## Part 4: break it (if you have time)

Change one thing at a time and look at the spectrogram again:

- `--seconds 1` and `--seconds 20`: what happens to the picture?
- `--fmin 100 --fmax 2000`: how does it sound now, and why?
- Play the WAV through your speakers, record it with your phone, and look
  at the recording. Does the picture survive the room?

## To hand in

One spectrogram PNG of your own secret, and two sentences: which window
size shows it best, and why.

## To discuss afterwards

- The picture is in the sound, but you cannot hear it. What does that say
  about how hearing and the Fourier transform differ?
- What would MP3 compression do to your picture?
- Where is this used for real? Think of watermarks and of Shazam.
