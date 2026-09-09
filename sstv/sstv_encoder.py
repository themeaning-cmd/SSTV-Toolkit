"""
sstv_encoder.py
Turns an image into an SSTV audio signal.

How an image becomes audio, in short:
  1. The image is resized/cropped to the exact pixel grid the chosen
     mode expects (e.g. 320x256 for Martin M1).
  2. It's split into the channels that mode transmits - either plain
     Green/Blue/Red rows (Martin, Scottie) or Luma/Chroma rows (PD,
     Robot 36).
  3. Each row's 0-255 values are mapped linearly onto a frequency
     between 1500Hz (0, "black") and 2300Hz (255, "white"): this is
     frequency modulation, no different in kind from an FM radio
     station, just carrying brightness instead of a song.
  4. Those per-row tones are stitched together with fixed sync pulses
     (1200Hz) and porches (1500Hz) that mark line/channel boundaries,
     following the exact timing template for the mode - see
     sstv_modes.py.
  5. A VIS header goes in front: a short, purely digital preamble
     (1900Hz leader tones plus 1100/1300Hz data bits) that spells out
     which mode is coming, so a receiver can configure itself
     automatically.
  6. The whole timeline of instantaneous frequencies is synthesized
     into one continuous waveform and written out as a WAV file - the
     kind of audio you could play into a transceiver's mic input, or
     straight into an SDR's TX audio, exactly as-is.
"""
import numpy as np
from PIL import Image

from sstv_common import (
    F_BLACK, F_SYNC, const_tone, scan_line_tones, fm_synthesize,
    encode_vis, write_wav,
)
from sstv_modes import get_mode, scan_channels


def _load_rgb(image, width, height, fit_mode='letterbox'):
    """
    Load `image` (a path or a PIL.Image) as a (height, width, 3)
    float64 array, at exactly width x height.

    A source image whose aspect ratio doesn't match the mode's is
    handled according to fit_mode:
      'letterbox' (default) - the whole image is kept: it's scaled to
                   fit within the target box and centered on a black
                   background to reach the exact size. Nothing is
                   cropped or distorted, so what comes out the other
                   end after decoding is the complete original image,
                   just with black bars on two edges if the
                   proportions didn't already match.
      'crop'     - center-cropped to the target aspect ratio before
                   resizing, so nothing is distorted, but whatever's
                   past the crop line on a mismatched image is lost.
      'stretch'  - resized directly to the exact target size. Keeps
                   every bit of the original content but distorts the
                   proportions if the aspect ratio didn't match.
    """
    img = Image.open(image) if isinstance(image, str) else image
    img = img.convert('RGB')
    src_w, src_h = img.size

    if fit_mode == 'stretch':
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    elif fit_mode == 'crop':
        target_ratio = width / height
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            new_w, new_h = int(round(src_h * target_ratio)), src_h
        else:
            new_w, new_h = src_w, int(round(src_w / target_ratio))
        left, top = (src_w - new_w) // 2, (src_h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    elif fit_mode == 'letterbox':
        scale = min(width / src_w, height / src_h)
        new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (width, height), (0, 0, 0))
        canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
        img = canvas

    else:
        raise ValueError(f"unknown fit_mode {fit_mode!r}; use 'letterbox', 'crop', or 'stretch'")

    return np.asarray(img, dtype=np.float64)


def _rgb_to_ycrcb(rgb):
    """Standard BT.601 RGB -> Y/Cr/Cb (all 0-255), full range."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return y, cr, cb


def _build_scan_family_video(rows_by_channel, mode, sample_rate):
    """Concatenate a whole mode's video (everything after the VIS
    header) for the 'GBR' and 'PD' template-driven families."""
    segments = []
    if mode['leading_sync_ms']:
        segments.append(const_tone(F_SYNC, mode['leading_sync_ms'] / 1000.0, sample_rate))
    for i in range(mode['tx_lines']):
        for kind, param, dur_ms in mode['template']:
            dur_s = dur_ms / 1000.0
            if kind == 'sync':
                segments.append(const_tone(F_SYNC, dur_s, sample_rate))
            elif kind == 'gap':
                segments.append(const_tone(F_BLACK, dur_s, sample_rate))
            elif kind == 'scan':
                segments.append(scan_line_tones(rows_by_channel[param][i], dur_s, sample_rate))
            else:
                raise ValueError(f"unknown template segment kind {kind!r}")
    return np.concatenate(segments) if segments else np.zeros(0)


def _build_robot36_video(y_rows, cr_pair_rows, cb_pair_rows, mode, sample_rate):
    """
    Robot 36's line order: for each image line, send its own Y, then
    ONE of the two shared chroma channels for its line-pair - Cr after
    even lines (separator tone at F_BLACK), Cb after odd lines
    (separator tone at F_WHITE). A receiver recovers full color once it
    has both lines of a pair.
    """
    sync_s = mode['sync_ms'] / 1000.0
    porch1_s = mode['porch1_ms'] / 1000.0
    y_s = mode['y_ms'] / 1000.0
    sep_s = mode['sep_ms'] / 1000.0
    porch2_s = mode['porch2_ms'] / 1000.0
    chroma_s = mode['chroma_ms'] / 1000.0
    segments = []
    for line in range(mode['height']):
        pair, even = line // 2, (line % 2 == 0)
        segments.append(const_tone(F_SYNC, sync_s, sample_rate))
        segments.append(const_tone(F_BLACK, porch1_s, sample_rate))
        segments.append(scan_line_tones(y_rows[line], y_s, sample_rate))
        segments.append(const_tone(F_BLACK if even else 2300.0, sep_s, sample_rate))
        segments.append(const_tone(F_BLACK, porch2_s, sample_rate))
        chroma_row = cr_pair_rows[pair] if even else cb_pair_rows[pair]
        segments.append(scan_line_tones(chroma_row, chroma_s, sample_rate))
    return np.concatenate(segments)


def encode_image_to_audio(image, mode_name, sample_rate=44100, fit_mode='letterbox'):
    """
    Encode `image` (a file path or a PIL.Image) for SSTV mode
    `mode_name` (e.g. "Martin M1", "scottie1", "Robot 36", "PD120" -
    see sstv_modes.list_modes() for all of them).

    fit_mode controls what happens when the image's aspect ratio
    doesn't match the mode's - see _load_rgb's docstring for the three
    options. Defaults to 'letterbox': nothing gets cropped off.

    Returns (sample_rate, audio) where `audio` is a float64 waveform
    in [-1, 1], ready for sstv_common.write_wav or your own playback.
    """
    mode = get_mode(mode_name)
    rgb = _load_rgb(image, mode['width'], mode['height'], fit_mode=fit_mode)
    header = encode_vis(mode['vis_code'], sample_rate)

    if mode['family'] == 'GBR':
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        video = _build_scan_family_video({'G': g, 'B': b, 'R': r}, mode, sample_rate)

    elif mode['family'] == 'PD':
        y, cr, cb = _rgb_to_ycrcb(rgb)
        y0, y1 = y[0::2], y[1::2]
        cr_pair = (cr[0::2] + cr[1::2]) / 2.0
        cb_pair = (cb[0::2] + cb[1::2]) / 2.0
        video = _build_scan_family_video(
            {'Y0': y0, 'Cr': cr_pair, 'Cb': cb_pair, 'Y1': y1}, mode, sample_rate)

    elif mode['family'] == 'ROBOT36':
        y, cr, cb = _rgb_to_ycrcb(rgb)
        cr_pair = (cr[0::2] + cr[1::2]) / 2.0
        cb_pair = (cb[0::2] + cb[1::2]) / 2.0
        video = _build_robot36_video(y, cr_pair, cb_pair, mode, sample_rate)

    else:
        raise ValueError(f"unknown mode family {mode['family']!r}")

    freq_timeline = np.concatenate([header, video])
    audio = fm_synthesize(freq_timeline, sample_rate)
    return sample_rate, audio


def encode_image_to_wav(image, mode_name, output_path, sample_rate=44100, fit_mode='letterbox'):
    """Same as encode_image_to_audio, but writes the result straight to
    a WAV file at `output_path` and returns that path."""
    sr, audio = encode_image_to_audio(image, mode_name, sample_rate=sample_rate, fit_mode=fit_mode)
    write_wav(output_path, sr, audio)
    return output_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("usage: python3 sstv_encoder.py <image> <mode> <output.wav>")
        print()
        from sstv_modes import list_modes
        print("available modes:")
        print(list_modes())
        sys.exit(1)
    path = encode_image_to_wav(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"wrote {path}")
