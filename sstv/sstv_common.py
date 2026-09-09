"""
sstv_common.py
Shared low-level building blocks used by both the SSTV encoder and
decoder: tone/frequency constants, FM synthesis, instantaneous-
frequency demodulation, VIS header encode/decode, and WAV I/O.

Every analog SSTV mode reduces to one idea: at any instant, the audio
is playing a single tone, and that tone's frequency IS the information
(a pixel brightness, a sync pulse, a header bit). Encoding is "decide
the frequency at every moment, then synthesize it." Decoding is
"measure the frequency at every moment, then interpret it." Everything
in this file supports one of those two directions.

Timing/frequency constants below follow J.L. Barber (N7CXI),
"Proposal for SSTV Mode Specifications" (Dayton SSTV Forum, 2000),
the de-facto reference most modern SSTV software is built against.
"""
import numpy as np
from scipy import signal
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Protocol-wide tone constants (Hz). Identical across every mode below.
F_SYNC = 1200.0     # horizontal sync pulse
F_BLACK = 1500.0    # video "black" level, also used for porches/separators
F_WHITE = 2300.0    # video "white" level
F_LEADER = 1900.0   # VIS calibration leader tone
F_VIS_0 = 1300.0    # VIS data bit = 0
F_VIS_1 = 1100.0    # VIS data bit = 1
VIS_BIT_S = 0.030   # duration of each VIS leader/data/parity/stop bit


# ---------------------------------------------------------------------------
# Pixel value <-> instantaneous frequency
def value_to_freq(value, f_lo=F_BLACK, f_hi=F_WHITE):
    """Map an 8-bit sample (0-255: luma, or a color channel) to its tone in Hz."""
    value = np.asarray(value, dtype=np.float64)
    return f_lo + (np.clip(value, 0.0, 255.0) / 255.0) * (f_hi - f_lo)


def freq_to_value(freq, f_lo=F_BLACK, f_hi=F_WHITE):
    """Inverse of value_to_freq: a demodulated tone back to a 0-255 sample."""
    v = (freq - f_lo) * 255.0 / (f_hi - f_lo)
    return np.clip(v, 0.0, 255.0)


# ---------------------------------------------------------------------------
# FM synthesis (encoder side)
def fm_synthesize(freq_hz, sample_rate, amplitude=0.9):
    """
    Phase-continuous FM synthesis. `freq_hz` is a 1D array giving the
    instantaneous frequency (Hz) at every audio sample; returns the
    waveform (float64, +-amplitude) of the same length.

    Phase continuity matters here: naively restarting sin(2*pi*f*t) at
    t=0 for every new segment creates a small phase jump at every
    boundary between tones. That jump sounds like a click and, more
    importantly, splatters broadband noise right at the moment a
    decoder most needs a clean, narrow tone to measure. Treating
    frequency as the derivative of phase and integrating it (a running
    sum, since we work in discrete samples) keeps phase unbroken across
    every segment boundary.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    phase = 2.0 * np.pi * np.cumsum(freq_hz) / sample_rate
    return amplitude * np.sin(phase)


def const_tone(freq_hz, duration_s, sample_rate):
    """`duration_s` seconds of a fixed frequency, as an instantaneous-
    frequency array (concatenate these to build up a full transmission)."""
    n = max(0, int(round(duration_s * sample_rate)))
    return np.full(n, float(freq_hz), dtype=np.float64)


def scan_line_tones(values, duration_s, sample_rate, f_lo=F_BLACK, f_hi=F_WHITE):
    """
    Turn one row of 0-255 samples into an instantaneous-frequency array
    spanning exactly `duration_s` seconds. Each sample gets an equal
    share of the time. Segment boundaries are computed from a running
    sample count (round(i * total_samples / n)) rather than by rounding
    each pixel's duration independently, so small roundings can't
    accumulate into a drift across a 320-pixel row.
    """
    values = np.asarray(values, dtype=np.float64)
    n_values = len(values)
    total_samples = int(round(duration_s * sample_rate))
    if n_values == 0 or total_samples <= 0:
        return np.zeros(max(total_samples, 0), dtype=np.float64)
    edges = np.round(np.arange(n_values + 1) * total_samples / n_values).astype(np.int64)
    freqs = value_to_freq(values, f_lo, f_hi)
    out = np.empty(total_samples, dtype=np.float64)
    for i in range(n_values):
        out[edges[i]:edges[i + 1]] = freqs[i]
    return out


# ---------------------------------------------------------------------------
# VIS header
def _vis_bits(vis_code):
    """7 data bits (LSB first) plus one even-parity bit for a VIS code."""
    bits = [(vis_code >> i) & 1 for i in range(7)]
    parity = sum(bits) % 2
    return bits + [parity]


def encode_vis(vis_code, sample_rate):
    """
    Instantaneous-frequency array for a complete VIS header:
    300ms leader / 10ms break / 300ms leader / 30ms start bit /
    7 data bits (LSB first) / 1 even-parity bit / 30ms stop bit.
    This is the only part of an SSTV transmission that carries
    digital (as opposed to continuously-varying) information, and it's
    what lets a receiver identify the mode automatically.
    """
    segs = [
        const_tone(F_LEADER, 0.300, sample_rate),
        const_tone(F_SYNC, 0.010, sample_rate),
        const_tone(F_LEADER, 0.300, sample_rate),
        const_tone(F_SYNC, VIS_BIT_S, sample_rate),  # start bit
    ]
    for bit in _vis_bits(vis_code):
        segs.append(const_tone(F_VIS_1 if bit else F_VIS_0, VIS_BIT_S, sample_rate))
    segs.append(const_tone(F_SYNC, VIS_BIT_S, sample_rate))  # stop bit
    return np.concatenate(segs)


# ---------------------------------------------------------------------------
# Decoder-side signal processing
def instantaneous_frequency(audio, sample_rate, band=(900.0, 2700.0)):
    """
    Band-limit the audio to the SSTV tone range and return its
    instantaneous frequency (Hz) at every sample, via a Hilbert
    transform. Every later decoding step (VIS bits, sync pulses, pixel
    values) reads from this one array: SSTV never carries information
    any other way than "what tone is playing right now."
    """
    audio = np.asarray(audio, dtype=np.float64)
    nyq = sample_rate / 2.0
    lo, hi = band[0] / nyq, min(band[1], nyq * 0.99) / nyq
    sos = signal.butter(6, [lo, hi], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, audio)
    analytic = signal.hilbert(filtered)
    phase = np.unwrap(np.angle(analytic))
    freq = np.diff(phase) * sample_rate / (2.0 * np.pi)
    freq = np.concatenate([freq[:1], freq])  # keep the same length as audio
    return signal.medfilt(freq, 7)  # squash isolated single-sample glitches


def _runs_below(freq, i0, i1, threshold):
    """[(start,end), ...] index pairs where freq[i0:i1] dips below
    `threshold` - used to find sync pulses and VIS bits, which are all
    "the tone dropped to around 1200 Hz for a while" events."""
    mask = freq[i0:i1] < threshold
    if not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return [(i0 + s, i0 + e) for s, e in zip(starts, ends)]


def _read_vis_bits(freq, sample_rate, vis_start, offset, shift_samples=0):
    """Read the 8 VIS bits (7 data + 1 parity) with the whole bit grid
    shifted by `shift_samples` from its nominal position relative to
    vis_start. Returns (bits, vis_code, parity_ok), or None if the
    window runs past the end of `freq`."""
    bits = []
    for b in range(8):
        # Bit b's window is centered (b+1) bit-lengths after the start
        # bit begins, offset by half a bit for the center.
        center = (vis_start + shift_samples
                  + int(round((VIS_BIT_S * (b + 1) + VIS_BIT_S / 2.0) * sample_rate)))
        half_win = int(round(0.008 * sample_rate))
        lo, hi = max(0, center - half_win), min(len(freq), center + half_win)
        if hi <= lo:
            return None
        avg = float(np.mean(freq[lo:hi])) - offset
        bits.append(1 if avg < (F_VIS_0 + F_VIS_1) / 2.0 else 0)
    vis_code = sum(bit << i for i, bit in enumerate(bits[:7]))
    parity_ok = (sum(bits[:7]) % 2) == bits[7]
    return bits, vis_code, parity_ok


def find_vis_header(freq, sample_rate, start=0, known_codes=None):
    """
    Scan freq[start:] for the next VIS header: a run of >=200ms near
    1900Hz (the leader), followed within a second by a >=20ms drop
    (the 1200Hz start bit), followed by 7 data bits + parity.

    `known_codes`, if given, is the set of VIS codes the caller can
    actually do anything with (sstv_decoder passes the mode registry's
    keys). It's used only as a validity check, not to change which
    header is found: a sharp tone transition like the leader-to-start-
    bit edge can, under resampling or real clock-rate mismatch, be
    detected a couple of milliseconds from where it truly is, and
    every bit window is anchored to that one edge, so a small miss
    there biases several bits at once - usually still harmless, but
    occasionally enough to flip one under real noise. If the reading
    at the nominal alignment doesn't both pass parity and land on a
    code the caller recognizes, a few microsecond-scale shifts of the
    whole bit grid are tried and kept only if a shift satisfies both
    of those independent, narrow constraints - something a genuine
    misread is unlikely to do by chance - so this can only recover a
    marginal signal, never talk a clean one into a wrong answer.

    Returns None, or a dict with:
      leader_index      - sample index where the leader tone began
      offset_hz         - how far every tone in this transmission is
                           shifted from nominal (a transmitter or
                           receiver dial not exactly on frequency);
                           later decoding subtracts this back out
      vis_code          - the decoded 7-bit mode code
      parity_ok         - whether the parity bit checked out
      header_end_index  - sample index where the header ends and the
                           actual picture (or a mode's own leading
                           sync, for modes that have one) begins
    """
    block = max(1, int(round(0.010 * sample_rate)))
    n_blocks = (len(freq) - start) // block
    if n_blocks < 25:
        return None
    means = freq[start:start + n_blocks * block].reshape(n_blocks, block).mean(axis=1)

    j = 0
    while j < n_blocks - 20:
        if not (1600.0 < means[j] < 2200.0):
            j += 1
            continue
        k = j
        while k < n_blocks and 1600.0 < means[k] < 2200.0 \
                and abs(means[k] - float(np.mean(means[j:k + 1]))) < 80.0:
            k += 1
        run_ms = (k - j) * 10.0
        if run_ms < 200.0:
            j = max(j + 1, k)
            continue
        offset = float(np.mean(means[j:k])) - F_LEADER
        leader_end = start + k * block
        # The start bit immediately follows the leader (only a ~10ms
        # break between them by protocol), so look there first with a
        # narrow window - generous enough to absorb real clock drift,
        # but tight enough that noise fragmenting the true start bit
        # can't cause this to skip past it and lock onto some later,
        # coincidentally-low-toned data bit instead (which, once
        # anchored on, would throw off every bit reading after it).
        # Only fall back to the original wide window if nothing turns
        # up close by at all.
        near_hi = min(len(freq), leader_end + int(round(0.08 * sample_rate)))
        drop = _first_run_at_least(freq, leader_end, near_hi, offset + 1250.0, 0.020, sample_rate)
        if drop is None:
            search_hi = min(len(freq), leader_end + sample_rate)
            drop = _first_run_at_least(freq, leader_end, search_hi, offset + 1250.0, 0.020, sample_rate)
        if drop is None:
            j = k
            continue
        vis_start = drop
        result = _read_vis_bits(freq, sample_rate, vis_start, offset)
        if result is None:
            j = k
            continue
        bits, vis_code, parity_ok = result

        good = parity_ok and (known_codes is None or vis_code in known_codes)
        if not good:
            # Defense in depth: if the wide fallback above (or anything
            # else) still locked onto the wrong dip, it will be some
            # whole number of bit-periods from the true start bit, by
            # construction - try correcting for that directly, negative
            # (earlier) shifts first since an overshoot is the only way
            # this search can go wrong (it returns the *first* qualifying
            # dip after the leader, so a false lock is always later than
            # the truth, never earlier).
            for shift_bits in (-1, -2, -3, -4, 1, 2, 3, 4):
                shift_samples = int(round(shift_bits * VIS_BIT_S * sample_rate))
                alt = _read_vis_bits(freq, sample_rate, vis_start, offset, shift_samples)
                if alt is None:
                    continue
                alt_bits, alt_code, alt_parity = alt
                if alt_parity and (known_codes is None or alt_code in known_codes):
                    bits, vis_code, parity_ok = alt_bits, alt_code, alt_parity
                    vis_start += shift_samples
                    break

        header_end = vis_start + int(round(10 * VIS_BIT_S * sample_rate))  # start+7+parity+stop
        return dict(
            leader_index=start + j * block,
            offset_hz=offset,
            vis_code=vis_code,
            parity_ok=parity_ok,
            header_end_index=header_end,
        )
    return None


def _first_run_at_least(freq, i0, i1, threshold, min_duration_s, sample_rate):
    """Index of the start of the first run in freq[i0:i1] that stays
    below `threshold` for at least `min_duration_s` seconds, or None."""
    min_len = int(round(min_duration_s * sample_rate))
    for s, e in _runs_below(freq, i0, i1, threshold):
        if e - s >= min_len:
            return s
    return None


# ---------------------------------------------------------------------------
# WAV I/O
def read_wav_mono(path):
    """Read a WAV file, returning (sample_rate, audio) as float64 in
    roughly [-1, 1], collapsing multi-channel audio to mono."""
    sample_rate, audio = wavfile.read(path)
    if audio.dtype.kind == 'i':
        audio = audio.astype(np.float64) / float(np.iinfo(audio.dtype).max)
    elif audio.dtype.kind == 'u':
        half = (np.iinfo(audio.dtype).max + 1) / 2.0
        audio = (audio.astype(np.float64) - half) / half
    else:
        audio = audio.astype(np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return sample_rate, audio


def write_wav(path, sample_rate, audio):
    """Write a float64 waveform (roughly [-1, 1]) out as 16-bit PCM WAV."""
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sample_rate, (clipped * 32767.0).astype(np.int16))
