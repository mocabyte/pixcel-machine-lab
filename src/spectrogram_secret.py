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

    # Each picture column becomes one short slice of time,
    # each picture row becomes one frequency.
    cols = int(ROWS * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (cols, ROWS), interpolation=cv2.INTER_AREA)
    loud = (img.astype(np.float64) / 255.0) ** 2          # brightness -> loudness

    # Top row of the picture = highest frequency.
    freqs = np.linspace(args.fmax, args.fmin, ROWS)
    per_col = int(RATE * args.seconds / cols)
    t = np.arange(cols * per_col) / RATE
    phase = np.random.default_rng(0).uniform(0, 2 * np.pi, ROWS)

    signal = np.zeros(cols * per_col)
    for c in range(cols):
        ts = t[c * per_col:(c + 1) * per_col]
        tones = np.sin(2 * np.pi * freqs[:, None] * ts[None, :] + phase[:, None])
        signal[c * per_col:(c + 1) * per_col] = loud[:, c] @ tones

    path = output_path(args.out)
    write_wav(path, signal)
    print(f"picture   {cols} x {ROWS} px")
    print(f"sound     {len(signal) / RATE:.1f} s, {args.fmin:.0f} to {args.fmax:.0f} Hz")
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
    top = int(args.fmax / (rate / 2) * (spec.shape[0] - 1))
    db = 20 * np.log10(spec[:top] + 1e-6)
    db = np.clip((db - db.max() + args.range) / args.range, 0, 1)
    pic = cv2.applyColorMap((db[::-1] * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    pic = cv2.resize(pic, (1200, 600), interpolation=cv2.INTER_AREA)

    path = output_path(Path(args.wav).stem + f"_spectrogram_w{win}.png")
    cv2.imwrite(str(path), pic)
    print(f"sound                 {len(signal) / rate:.1f} s at {rate} Hz")
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
    h.add_argument("--fmin", type=float, default=1000.0, help="lowest frequency in Hz")
    h.add_argument("--fmax", type=float, default=9000.0, help="highest frequency in Hz")
    h.set_defaults(func=hide)

    s = sub.add_parser("show", help="draw the spectrogram of a WAV file")
    s.add_argument("--wav", required=True)
    s.add_argument("--window", type=int, default=1024, help="FFT window in samples")
    s.add_argument("--fmax", type=float, default=10000.0, help="highest frequency shown")
    s.add_argument("--range", type=float, default=60.0, help="dynamic range in dB")
    s.set_defaults(func=show)

    args = ap.parse_args()
    if args.cmd == "hide" and not (args.image or args.text):
        ap.error("hide needs --image or --text")
    args.func(args)


if __name__ == "__main__":
    main()
