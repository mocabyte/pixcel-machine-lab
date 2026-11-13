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

## Part 5: hide it in your own sound

This part came from the group on 13 November: use your own audio and hide
a text in it. That is how audio watermarks work.

1. Pick a carrier: a song, a voice memo, or `data/output.wav` from the
   repo. WAV works directly. MP3 works if `librosa` is installed.

2. Mix your secret into it:

   ```
   python spectrogram_secret.py hide --text "HELLO" --carrier ../data/output.wav --out mixed.wav
   ```

   The script prints which frequency band it used for the secret. It
   picks a high band, where music has little energy.

3. Play `output/mixed.wav`. Can you hear that something was added?

4. Look at the whole spectrogram, then zoom into the band the script
   printed (the numbers below fit `data/output.wav`):

   ```
   python spectrogram_secret.py show --wav ../output/mixed.wav
   python spectrogram_secret.py show --wav ../output/mixed.wav --fmin 7000 --fmax 10500
   ```

5. Now make it quieter. `--level` is the loudness of the secret next to
   your sound, in dB. Try -10, -20, -40 and -60:

   ```
   python spectrogram_secret.py hide --text "HELLO" --carrier ../data/output.wav --level -40 --out quiet.wav
   ```

   Find the level where you can no longer hear the secret but can still
   read it. Then find the level where it disappears from the picture too.

6. Move it. `--fmin 300 --fmax 3000` puts the secret in the middle of the
   music. What happens to the sound, and to the picture?

Other options: `--start 10` begins the secret at second 10, `--seconds 3`
makes it shorter.

## To hand in

One spectrogram PNG of your own secret, and two sentences: which window
size shows it best, and why. If you did Part 5: the quietest level at which
your secret is still readable.

## To discuss afterwards

- The picture is in the sound, but you cannot hear it. What does that say
  about how hearing and the Fourier transform differ?
- What would MP3 compression do to your picture?
- Where is this used for real? Think of watermarks and of Shazam.
- Save your mixed sound as MP3 and look again. Is the secret still there?
  Why would a high, quiet secret be the first thing MP3 removes?
