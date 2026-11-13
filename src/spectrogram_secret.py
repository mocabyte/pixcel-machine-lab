"""
spectrogram_secret.py — hide a picture in a sound, and find it again.

A spectrogram shows which frequencies are in a sound over time. So if we
build a sound whose frequencies follow the pixels of a picture, the picture
appears in the spectrogram. You cannot see it in the waveform and you can
hardly hear it, but the Fourier transform finds it.

Run from the src/ folder:

  1. Hide a picture or a text in a sound
      python spectrogram_secret.py hide --image ../data/dog.jpg --out secret.wav
      python spectrogram_secret.py hide --text "HELLO" --out secret.wav

  2. Look at the spectrogram of any WAV file
      python spectrogram_secret.py show --wav ../output/secret.wav

  3. Change the window and look again
      python spectrogram_secret.py show --wav ../output/secret.wav --window 256
      python spectrogram_secret.py show --wav ../output/secret.wav --window 4096

  4. Hide the secret inside your own sound (a song, a voice recording)
      python spectrogram_secret.py hide --text "HELLO" --carrier my_song.wav --out mixed.wav
      python spectrogram_secret.py show --wav ../output/mixed.wav
      python spectrogram_secret.py show --wav ../output/mixed.wav --fmin 13000 --fmax 19000

     --level sets how loud the secret is next to your sound, in dB.
     -10 is easy to see and easy to hear, -40 is hidden well.

Needs only numpy and opencv.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import cv2
import numpy as np

from common import output_path

RATE = 22050          # samples per second
ROWS = 160            # picture height = number of frequencies we use


# --------------------------------------------------------------------------
# WAV in and out (16-bit mono), with Python's built-in wave module
# --------------------------------------------------------------------------
def write_wav(path: Path, signal: np.ndarray, rate: int = RATE) -> None:
    signal = signal / (np.abs(signal).max() + 1e-9) * 0.9
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((signal * 32767).astype(np.int16).tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if path.suffix.lower() != ".wav":
        # MP3 and friends: use librosa if it is installed.
        try:
            import librosa
        except ImportError:
            raise SystemExit(
                f"{path.name} is not a WAV file. Convert it to WAV first, "
                "or run 'pip install librosa soundfile' and try again."
            )
        signal, rate = librosa.load(str(path), sr=None, mono=True)
        return signal.astype(np.float64), int(rate)
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise SystemExit("Only 16-bit WAV files are supported.")
        rate, channels = f.getframerate(), f.getnchannels()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    signal = data.reshape(-1, channels).mean(axis=1) / 32768.0
    return signal, rate


# --------------------------------------------------------------------------
# hide: picture -> sound
# --------------------------------------------------------------------------
def text_image(text: str) -> np.ndarray:
    """Render a text as a white-on-black picture."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 3.0, 8
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    img = np.zeros((h + base + 40, w + 40), np.uint8)
    cv2.putText(img, text, (20, h + 20), font, scale, 255, thick, cv2.LINE_AA)
    return img


def hide(args):
    if args.text:
        img = text_image(args.text)
    else:
        img = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"Could not read image {args.image}")

    # With a carrier, the secret is mixed into your own sound. It goes high
    # up, where music has little energy and ears are weak.
    carrier, rate = None, RATE
    if args.carrier:
        carrier, rate = read_wav(Path(args.carrier))
    fmin = args.fmin if args.fmin else (14000.0 if carrier is not None else 1000.0)
    fmax = args.fmax if args.fmax else (18000.0 if carrier is not None else 9000.0)
    limit = 0.45 * rate
    if fmax > limit:
        fmin, fmax = fmin * limit / fmax, limit
        print(f"Your sound has {rate} samples per second, so it cannot hold "
              f"frequencies above {rate / 2:.0f} Hz. Using {fmin:.0f} to {fmax:.0f} Hz.\n")
    seconds = args.seconds
    if carrier is not None:
        seconds = min(seconds, len(carrier) / rate - args.start)
        if seconds <= 0.5:
            raise SystemExit("The carrier is too short for --start and --seconds.")

    # Each picture column becomes one short slice of time,
    # each picture row becomes one frequency.
    cols = int(ROWS * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (cols, ROWS), interpolation=cv2.INTER_AREA)
    loud = (img.astype(np.float64) / 255.0) ** 2          # brightness -> loudness

    # Top row of the picture = highest frequency.
    freqs = np.linspace(fmax, fmin, ROWS)
    per_col = max(int(rate * seconds / cols), 1)
    t = np.arange(cols * per_col) / rate
    phase = np.random.default_rng(0).uniform(0, 2 * np.pi, ROWS)

    signal = np.zeros(cols * per_col)
    for c in range(cols):
        ts = t[c * per_col:(c + 1) * per_col]
        tones = np.sin(2 * np.pi * freqs[:, None] * ts[None, :] + phase[:, None])
        signal[c * per_col:(c + 1) * per_col] = loud[:, c] @ tones

    path = output_path(args.out)
    print(f"picture   {cols} x {ROWS} px")
    if carrier is None:
        write_wav(path, signal, rate)
        print(f"sound     {len(signal) / rate:.1f} s, {fmin:.0f} to {fmax:.0f} Hz")
    else:
        # Set the loudness of the secret relative to the carrier, then add.
        a = int(args.start * rate)
        signal = signal[:len(carrier) - a]
        rms = lambda x: np.sqrt(np.mean(x ** 2)) + 1e-12
        gain = rms(carrier[a:a + len(signal)]) / rms(signal) * 10 ** (args.level / 20)
        mixed = carrier.copy()
        mixed[a:a + len(signal)] += gain * signal
        write_wav(path, mixed, rate)
        print(f"carrier   {Path(args.carrier).name}, {len(carrier) / rate:.1f} s at {rate} Hz")
        print(f"secret    {len(signal) / rate:.1f} s from second {args.start:.1f}, "
              f"{fmin:.0f} to {fmax:.0f} Hz, {args.level:.0f} dB next to the carrier")
    print(f"written -> {path}")
    print("Play it. Then run 'show' on it.")


# --------------------------------------------------------------------------
# show: sound -> spectrogram
# --------------------------------------------------------------------------
def show(args):
    signal, rate = read_wav(Path(args.wav))
    win, hop = args.window, max(args.window // 4, 1)
    if len(signal) < win:
        raise SystemExit("The sound is shorter than one window.")

    # Short-time Fourier transform: cut the sound into overlapping windows
    # and take the FFT of each one. Every FFT is one column of the picture.
    starts = np.arange(0, len(signal) - win, hop)
    frames = np.stack([signal[s:s + win] for s in starts]) * np.hanning(win)
    spec = np.abs(np.fft.rfft(frames, axis=1)).T          # rows = frequency

    # Keep the range we care about, convert to decibels, low notes at the bottom.
    hz = rate / 2 / (spec.shape[0] - 1)
    fmax = min(args.fmax or rate / 2, rate / 2)
    bottom, top = int(args.fmin / hz), int(fmax / hz)
    if top - bottom < 2:
        raise SystemExit("--fmin and --fmax leave nothing to show.")
    db = 20 * np.log10(spec[bottom:top] + 1e-6)
    db = np.clip((db - db.max() + args.range) / args.range, 0, 1)
    pic = cv2.applyColorMap((db[::-1] * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    pic = cv2.resize(pic, (1200, 600), interpolation=cv2.INTER_AREA)

    band = f"_{args.fmin:.0f}-{fmax:.0f}Hz" if (args.fmin or args.fmax) else ""
    path = output_path(Path(args.wav).stem + f"_spectrogram_w{win}{band}.png")
    cv2.imwrite(str(path), pic)
    print(f"sound                 {len(signal) / rate:.1f} s at {rate} Hz")
    print(f"shown                 {args.fmin:.0f} to {fmax:.0f} Hz")
    print(f"window                {win} samples = {win / rate * 1000:.0f} ms")
    print(f"frequency resolution  {rate / win:.1f} Hz per row")
    print(f"time resolution       {hop / rate * 1000:.1f} ms per column")
    print(f"written -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hide", help="turn a picture or a text into a sound")
    h.add_argument("--image", help="picture to hide")
    h.add_argument("--text", help="text to hide instead of a picture")
    h.add_argument("--out", default="secret.wav")
    h.add_argument("--seconds", type=float, default=6.0, help="length of the sound")
    h.add_argument("--fmin", type=float, default=None, help="lowest frequency in Hz (default 1000, with a carrier 14000)")
    h.add_argument("--fmax", type=float, default=None, help="highest frequency in Hz (default 9000, with a carrier 18000)")
    h.add_argument("--carrier", help="your own sound to hide the secret in (WAV, or MP3 if librosa is installed)")
    h.add_argument("--level", type=float, default=-20.0, help="loudness of the secret next to the carrier, in dB")
    h.add_argument("--start", type=float, default=0.0, help="second of the carrier where the secret begins")
    h.set_defaults(func=hide)

    s = sub.add_parser("show", help="draw the spectrogram of a WAV file")
    s.add_argument("--wav", required=True)
    s.add_argument("--window", type=int, default=1024, help="FFT window in samples")
    s.add_argument("--fmin", type=float, default=0.0, help="lowest frequency shown")
    s.add_argument("--fmax", type=float, default=None, help="highest frequency shown (default: everything)")
    s.add_argument("--range", type=float, default=60.0, help="dynamic range in dB")
    s.set_defaults(func=show)

    args = ap.parse_args()
    if args.cmd == "hide" and not (args.image or args.text):
        ap.error("hide needs --image or --text")
    args.func(args)


if __name__ == "__main__":
    main()
