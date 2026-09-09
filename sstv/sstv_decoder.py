"""
sstv_decoder.py
Turns SSTV audio back into an image, and can automatically find and
identify SSTV transmissions in a recording without being told a mode
in advance.

Decoding, in short, undoes the encoder step by step:
  1. The audio is reduced to one number per sample: its instantaneous
     frequency (sstv_common.instantaneous_frequency). This is the
     signal everything below reads from.
  2. We look for a VIS header - a sustained ~1900Hz tone followed by a
     short burst of 1100/1300Hz bits - and decode which mode is
     coming, purely from the audio itself (`find_vis_header`,
     `decode_wav_to_image` with mode_name=None). This is the
     "automatic waveform detection" half of this module: no prior
     knowledge of the mode, or even of where in the file the
     transmission starts, is required.
  3. Knowing the mode's exact timing template, we locate every line's
     sync pulse (a dip to ~1200Hz), and fit a straight line through
     all of them. The fitted line's slope is the *measured* line
     duration; if it differs slightly from the mode's nominal value,
     that's a sound-card or SDR clock running a little fast or slow
     over the whole transmission, and every pixel's sampling window is
     stretched by that same ratio to compensate ("slant correction").
  4. Each pixel's window of instantaneous frequency is averaged (via a
     cumulative-sum trick, so this is O(1) per pixel rather than a
     fresh average each time) and mapped back from Hz to a 0-255 value
     - the exact inverse of the encoder's mapping.
  5. Channels are recombined into RGB (YCrCb -> RGB for the PD and
     Robot 36 families) and returned as a Pillow Image.

`scan_for_transmissions` repeats steps 2-5 in a loop across an entire
recording, so a long capture (e.g. monitoring a repeater or a net for
a while) can be handed over untouched and every picture in it found
and decoded on its own.
"""
import numpy as np
from PIL import Image

from sstv_common import (
    F_SYNC, F_BLACK, F_WHITE, instantaneous_frequency, find_vis_header,
    freq_to_value, read_wav_mono, _runs_below,
)
from sstv_modes import MODES, get_mode, sync_offset_ms, scan_channels


# ---------------------------------------------------------------------------
# Shared sync-fitting: locate every transmission line's sync pulse and
# fit center = a + b*i (i = line index) by least squares, with one
# round of outlier rejection. `b` is the *measured* line period.
#
# A plain "search near i * nominal_period for every line" breaks down
# over a long transmission: a sound card or SDR clock even slightly off
# nominal (a fraction of a percent) accumulates into a drift many times
# a search window's width by the last lines, and they're never found.
# So acquisition is sequential instead - each line's search is centered
# on a prediction built from the *actually observed* period of the last
# couple of lines, which only ever has to absorb one line's worth of
# drift plus jitter, however far the whole transmission has drifted by
# that point. The final (a, b) is then smoothed with a global fit.
def _fit_line_centers(freq, sample_rate, a0, nominal_period, n_lines, sync_ms, offset_hz,
                       floor_index=0):
    """
    `floor_index` is the earliest sample a genuine line-0 sync could
    possibly start at (right after the VIS header, plus any mode
    leading sync). It matters for modes where the sync sits at the very
    start of the line template (Martin, PD): there, line 0's sync
    pulse is immediately preceded by the VIS header's own stop bit,
    which is *also* a ~1200Hz dip. Without a floor, a search window
    that reaches back far enough can merge the stop bit and the true
    sync pulse into one long sub-threshold run and report a falsely
    early, wrong center. Clamping every window to floor_index rules
    that out structurally instead of hoping the window is narrow
    enough to miss it.
    """
    thresh = offset_hz + (F_SYNC + F_BLACK) / 2.0
    sync_len = (sync_ms / 1000.0) * sample_rate

    def find_one(center_guess, win):
        i0 = max(int(center_guess - win), int(floor_index))
        i1 = int(center_guess + win)
        if i1 <= i0 or i1 > len(freq):
            return None
        runs = [(s, e) for s, e in _runs_below(freq, i0, i1, thresh)
                if (e - s) >= 0.5 * sync_len]
        if not runs:
            return None
        s, e = max(runs, key=lambda r: r[1] - r[0])
        return (s + e) / 2.0

    wide_win = max(0.1 * nominal_period, 0.015 * sample_rate)
    narrow_win = max(0.03 * nominal_period, 0.006 * sample_rate)

    # Bootstrap: check several early lines independently against the
    # nominal model (drift can't have accumulated far yet this early)
    # and fit a line through whichever are mutually consistent. This
    # needs several agreeing points before trusting a period estimate,
    # rather than risking everything on one detection that might be a
    # false positive with nothing yet to cross-check it against.
    boot_n = min(n_lines, 12)
    boot_pts = []
    for i in range(boot_n):
        c = find_one(a0 + i * nominal_period, wide_win)
        if c is not None:
            boot_pts.append((i, c))

    if len(boot_pts) < 3:
        return a0, nominal_period, len(boot_pts), False

    xs = np.array([p[0] for p in boot_pts], dtype=np.float64)
    ys = np.array([p[1] for p in boot_pts], dtype=np.float64)
    period, a_fit = np.polyfit(xs, ys, 1)
    keep = np.abs(ys - (a_fit + period * xs)) < 0.3 * nominal_period
    if keep.sum() >= 3:
        xs, ys = xs[keep], ys[keep]
        period, a_fit = np.polyfit(xs, ys, 1)
    pts = sorted(set(zip(xs.astype(int).tolist(), ys.tolist())))

    # Sequential tracking for the rest: each next line is searched only
    # around a prediction built from the most recently observed period,
    # so the window only has to absorb one line's worth of drift plus
    # jitter, however far the whole transmission has drifted by then.
    last_i, last_c = pts[-1]
    guess = last_c + period
    for i in range(last_i + 1, n_lines):
        c = find_one(guess, narrow_win)
        if c is not None:
            pts.append((i, c))
            period = c - last_c if i == last_i + 1 else (c - last_c) / (i - last_i)
            last_i, last_c = i, c
            guess = c + period
        else:
            guess = guess + period  # coast through a miss on the same track

    min_pts = max(8, int(0.2 * n_lines))
    if len(pts) < min_pts:
        return a0, nominal_period, len(pts), False

    # One global fit for a smooth, noise-averaged result, with outliers
    # (a sync mistaken for another dip in the image content) dropped.
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    b, a = np.polyfit(xs, ys, 1)
    keep = np.abs(ys - (a + b * xs)) < 0.005 * sample_rate
    xs, ys = xs[keep], ys[keep]
    if len(xs) < min_pts:
        return a0, nominal_period, len(xs), False
    b, a = np.polyfit(xs, ys, 1)
    return a, b, len(xs), True


def _cumsum_table(freq):
    """So that mean(freq[s:e]) == (table[e]-table[s])/(e-s) in O(1)."""
    return np.concatenate([[0.0], np.cumsum(freq)])


def _sample_row(cs, freq_len, t0, dur_samples, width, offset_hz):
    """Average `width` equal slices of [t0, t0+dur_samples) and map
    each from Hz back to a 0-255 value."""
    step = dur_samples / width
    starts = t0 + np.arange(width) * step
    ends = starts + step
    s = np.clip(starts.astype(np.int64), 0, freq_len - 1)
    e = np.clip(np.maximum(ends.astype(np.int64), s + 1), 1, freq_len)
    fmean = (cs[e] - cs[s]) / (e - s)
    return freq_to_value(fmean - offset_hz)


# ---------------------------------------------------------------------------
# 'GBR' and 'PD' families share one template-driven decoder.
def _decode_scan_family(freq, sample_rate, hdr, mode):
    offset = hdr['offset_hz']
    lead_samples = (mode['leading_sync_ms'] / 1000.0) * sample_rate
    origin0 = hdr['header_end_index'] + lead_samples

    sync_pos_ms = sync_offset_ms(mode)
    sync_ms = next(d for k, _, d in mode['template'] if k == 'sync')
    T_nom = (mode['line_ms'] / 1000.0) * sample_rate
    n = mode['tx_lines']

    a0 = origin0 + ((sync_pos_ms + sync_ms / 2.0) / 1000.0) * sample_rate
    a, b, nsync, ok = _fit_line_centers(freq, sample_rate, a0, T_nom, n, sync_ms, offset,
                                         floor_index=origin0)
    scale = b / T_nom
    note = (f"{nsync}/{n} lines aligned via sync detection" if ok else
            f"sync fit unreliable ({nsync} lines matched) - used nominal line timing")

    cs = _cumsum_table(freq)
    width = mode['width']
    chans = scan_channels(mode)
    rows = {c: np.zeros((n, width)) for c in chans}
    cum = []
    t = 0.0
    for kind, param, dur in mode['template']:
        cum.append((kind, param, t, dur))
        t += dur

    for i in range(n):
        sync_center = a + b * i
        line_origin = sync_center - ((sync_pos_ms + sync_ms / 2.0) / 1000.0) * sample_rate * scale
        for kind, param, start_ms, dur_ms in cum:
            if kind != 'scan':
                continue
            t0 = line_origin + (start_ms / 1000.0) * sample_rate * scale
            dur_samples = (dur_ms / 1000.0) * sample_rate * scale
            rows[param][i] = _sample_row(cs, len(freq), t0, dur_samples, width, offset)

    info = dict(note=note, measured_line_ms=b / sample_rate * 1000.0,
                slant_pct=(b - T_nom) / T_nom * 100.0, lines_synced=nsync)
    return rows, info


def _ycrcb_to_rgb(y, cr, cb):
    cr = cr - 128.0
    cb = cb - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1)


def _assemble_gbr_image(rows, mode):
    img = np.zeros((mode['height'], mode['width'], 3), dtype=np.float64)
    img[..., 0], img[..., 1], img[..., 2] = rows['R'], rows['G'], rows['B']
    return img


def _assemble_pd_image(rows, mode):
    img = np.zeros((mode['height'], mode['width'], 3), dtype=np.float64)
    img[0::2] = _ycrcb_to_rgb(rows['Y0'], rows['Cr'], rows['Cb'])
    img[1::2] = _ycrcb_to_rgb(rows['Y1'], rows['Cr'], rows['Cb'])
    return img


# ---------------------------------------------------------------------------
# Robot 36: bespoke decoder for its interleaved-chroma line structure.
def _decode_robot36(freq, sample_rate, hdr, mode):
    offset = hdr['offset_hz']
    origin0 = hdr['header_end_index']
    n = mode['height']  # one transmission slot per image line
    sync_ms, porch1_ms, y_ms = mode['sync_ms'], mode['porch1_ms'], mode['y_ms']
    sep_ms, porch2_ms, chroma_ms = mode['sep_ms'], mode['porch2_ms'], mode['chroma_ms']
    T_nom = (mode['line_ms'] / 1000.0) * sample_rate
    width = mode['width']

    a0 = origin0 + (sync_ms / 2.0 / 1000.0) * sample_rate
    a, b, nsync, ok = _fit_line_centers(freq, sample_rate, a0, T_nom, n, sync_ms, offset,
                                         floor_index=origin0)
    scale = b / T_nom
    note = (f"{nsync}/{n} lines aligned via sync detection" if ok else
            f"sync fit unreliable ({nsync} lines matched) - used nominal line timing")

    cs = _cumsum_table(freq)
    y_all = np.zeros((n, width))
    chroma_all = np.zeros((n, width))
    sep_freq = np.zeros(n)

    for i in range(n):
        sync_center = a + b * i
        line_origin = sync_center - (sync_ms / 2.0 / 1000.0) * sample_rate * scale

        y_start = line_origin + ((sync_ms + porch1_ms) / 1000.0) * sample_rate * scale
        y_dur = (y_ms / 1000.0) * sample_rate * scale
        y_all[i] = _sample_row(cs, len(freq), y_start, y_dur, width, offset)

        sep_start = y_start + y_dur
        sep_dur = (sep_ms / 1000.0) * sample_rate * scale
        ss = int(np.clip(sep_start, 0, len(freq) - 1))
        se = int(np.clip(max(sep_start + sep_dur, ss + 1), 1, len(freq)))
        sep_freq[i] = (cs[se] - cs[ss]) / (se - ss) - offset

        chroma_start = sep_start + sep_dur + (porch2_ms / 1000.0) * sample_rate * scale
        chroma_dur = (chroma_ms / 1000.0) * sample_rate * scale
        chroma_all[i] = _sample_row(cs, len(freq), chroma_start, chroma_dur, width, offset)

    # The separator's tone (not its position) says which chroma channel
    # a line carried: near F_BLACK -> Cr, near F_WHITE -> Cb.
    is_cr = np.abs(sep_freq - F_BLACK) < np.abs(sep_freq - F_WHITE)

    img = np.zeros((n, width, 3), dtype=np.float64)
    for pair in range(n // 2):
        i0, i1 = 2 * pair, 2 * pair + 1
        cr_row = chroma_all[i0] if is_cr[i0] else chroma_all[i1]
        cb_row = chroma_all[i1] if not is_cr[i1] else chroma_all[i0]
        img[i0] = _ycrcb_to_rgb(y_all[i0], cr_row, cb_row)
        img[i1] = _ycrcb_to_rgb(y_all[i1], cr_row, cb_row)

    info = dict(note=note, measured_line_ms=b / sample_rate * 1000.0,
                slant_pct=(b - T_nom) / T_nom * 100.0, lines_synced=nsync)
    return img, info


# ---------------------------------------------------------------------------
def _decode_with_mode(freq, sample_rate, hdr, mode):
    if mode['family'] == 'GBR':
        rows, info = _decode_scan_family(freq, sample_rate, hdr, mode)
        return _assemble_gbr_image(rows, mode), info
    if mode['family'] == 'PD':
        rows, info = _decode_scan_family(freq, sample_rate, hdr, mode)
        return _assemble_pd_image(rows, mode), info
    if mode['family'] == 'ROBOT36':
        return _decode_robot36(freq, sample_rate, hdr, mode)
    raise ValueError(f"unsupported mode family {mode['family']!r}")


def _mode_video_samples(mode, sample_rate):
    """Total sample count from the end of the VIS header to the end of
    this mode's picture - used to know how far to skip ahead when
    scanning a long recording for more than one transmission."""
    if mode['family'] == 'ROBOT36':
        return int(round((mode['height'] * mode['line_ms']) / 1000.0 * sample_rate))
    return int(round((mode['leading_sync_ms'] + mode['tx_lines'] * mode['line_ms'])
                      / 1000.0 * sample_rate))


def decode_wav_to_image(wav_path, output_image_path=None, mode_name=None):
    """
    Decode the first SSTV transmission in `wav_path` into a picture.

    If `mode_name` is None (the default), the mode is auto-detected
    from the VIS header - this is the normal way to use this function
    for a recording you haven't looked at yet. Pass a mode explicitly
    to force it (used automatically as a fallback if a VIS header
    can't be found at all, e.g. for audio that's been trimmed to just
    the picture).

    Returns (PIL.Image, info) where `info` describes what was found:
    mode name, VIS code, whether the VIS parity check passed, the
    measured frequency offset, and the line-sync fit quality.
    """
    sample_rate, audio = read_wav_mono(wav_path)
    freq = instantaneous_frequency(audio, sample_rate)

    hdr = find_vis_header(freq, sample_rate, 0, known_codes=MODES.keys())
    if hdr is not None:
        mode = get_mode(mode_name) if mode_name is not None else MODES.get(hdr['vis_code'])
        if mode is None:
            raise ValueError(
                f"Found a VIS header (code {hdr['vis_code']}) but it isn't one of the "
                f"modes this decoder supports. See sstv_modes.list_modes().")
    else:
        if mode_name is None:
            raise ValueError(
                "No SSTV VIS header found in this recording. If this audio has been "
                "trimmed to start right at the picture (no header), pass mode_name= "
                "explicitly.")
        mode = get_mode(mode_name)
        hdr = dict(leader_index=0, offset_hz=0.0, vis_code=mode['vis_code'],
                   parity_ok=True, header_end_index=0)

    img_arr, info = _decode_with_mode(freq, sample_rate, hdr, mode)
    img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8), 'RGB')
    if output_image_path:
        img.save(output_image_path)

    info.update(mode=mode['name'], vis_code=hdr['vis_code'],
                parity_ok=hdr['parity_ok'], frequency_offset_hz=hdr['offset_hz'],
                start_time_s=hdr['leader_index'] / sample_rate)
    return img, info


def scan_for_transmissions(wav_path, verbose=True):
    """
    Slide through an entire recording (e.g. a long monitoring capture
    of an SSTV net or repeater) and find every transmission in it,
    without any prior knowledge of when they start or what mode each
    one uses. This is the "detect radio waveforms automatically" entry
    point for unattended/bulk use.

    Returns a list of dicts, one per transmission found:
      time_s, vis_code, parity_ok, mode (name, or None if the VIS code
      isn't recognized), image (a PIL.Image, or None if it couldn't be
      decoded), info (the same dict decode_wav_to_image returns, if
      decoding succeeded)
    """
    sample_rate, audio = read_wav_mono(wav_path)
    freq = instantaneous_frequency(audio, sample_rate)

    results = []
    pos = 0
    while True:
        hdr = find_vis_header(freq, sample_rate, pos, known_codes=MODES.keys())
        if hdr is None:
            break
        t = hdr['leader_index'] / sample_rate
        mode = MODES.get(hdr['vis_code'])
        entry = dict(time_s=t, vis_code=hdr['vis_code'], parity_ok=hdr['parity_ok'],
                     mode=mode['name'] if mode else None, image=None, info=None)

        if mode is None:
            if verbose:
                print(f"t={t:7.1f}s  VIS={hdr['vis_code']:<3} not a recognized mode - skipped")
            pos = hdr['header_end_index'] + sample_rate  # hop past it and keep scanning
        else:
            try:
                img_arr, info = _decode_with_mode(freq, sample_rate, hdr, mode)
                entry['image'] = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8), 'RGB')
                entry['info'] = info
                if verbose:
                    print(f"t={t:7.1f}s  {mode['name']:<12} decoded  ({info['note']})")
            except Exception as ex:  # noqa: BLE001 - keep scanning past a bad transmission
                if verbose:
                    print(f"t={t:7.1f}s  {mode['name']:<12} found but failed to decode: {ex}")
            pos = hdr['header_end_index'] + _mode_video_samples(mode, sample_rate)
        results.append(entry)
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 sstv_decoder.py <input.wav> [output.png] [mode]")
        sys.exit(1)
    wav = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    forced_mode = sys.argv[3] if len(sys.argv) > 3 else None
    image, info = decode_wav_to_image(wav, out, mode_name=forced_mode)
    for k, v in info.items():
        print(f"{k}: {v}")
    if out:
        print(f"wrote {out}")
