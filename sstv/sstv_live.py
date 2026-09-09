"""
sstv_live.py
Live microphone monitoring. Continuously listens on the system's
input device, keeps a scrolling spectrogram of what's coming in, and
automatically detects and decodes an SSTV transmission the moment its
VIS header shows up - the live equivalent of scan_for_transmissions()
running on a growing buffer instead of a finished file.

This deliberately does *not* try to clean up what the microphone
picks up beyond what sstv_decoder.py already does for any recording.
A casual capture (a laptop mic held near a speaker, say) keeps
whatever room noise, echo, and sample-clock wobble came with it, and
that shows up honestly in the decoded image rather than being
smoothed away.
"""
import os
import time
import wave
import tempfile
import threading

import numpy as np
import sounddevice as sd
from PIL import Image

from sstv_common import instantaneous_frequency, find_vis_header
from sstv_modes import MODES, scan_channels
from sstv_decoder import decode_wav_to_image, _cumsum_table, _sample_row, _ycrcb_to_rgb

SAMPLE_RATE = 44100
VIS_CHECK_INTERVAL_S = 1.0   # how often to scan recent audio for a VIS header
VIS_CHECK_WINDOW_S = 4.0     # how much recent audio each scan looks at
IDLE_BUFFER_CAP_S = VIS_CHECK_WINDOW_S * 4  # trim the buffer to this while idle
WATERFALL_BINS = 110         # frequency resolution of the visual (not the decoder)
WATERFALL_COLS = 260         # scrolling history length
WATERFALL_LO_HZ, WATERFALL_HI_HZ = 900, 2500  # covers VIS bits through white
PREVIEW_MIN_INTERVAL_S = 1.5  # don't recompute the in-progress preview more often than this


def _quick_preview_image(buf, sample_rate, tx_start_sample, mode):
    """
    A fast, rough decode of whatever part of the current transmission
    has arrived so far, using plain nominal per-line timing - no sync
    fitting, no drift correction, none of the offset re-measurement
    the real decoder does. This is only ever shown while a
    transmission is still coming in, purely so there's a picture
    building up instead of a bare percentage; the moment the full
    transmission is captured, the real decode_wav_to_image (unchanged,
    the same one used everywhere else in this project) replaces it
    with the properly corrected result.

    Returns (PIL.Image, lines_drawn), or None if not even one line's
    worth of audio has arrived yet.
    """
    width, height = mode['width'], mode['height']
    available_s = (len(buf) - tx_start_sample) / sample_rate
    if available_s <= 0.05:
        return None

    segment = buf[tx_start_sample: tx_start_sample + int(available_s * sample_rate) + 1]
    freq = instantaneous_frequency(segment, sample_rate)
    cs = _cumsum_table(freq)
    img = np.zeros((height, width, 3), dtype=np.float64)

    if mode['family'] in ('GBR', 'PD'):
        lead_s = mode['leading_sync_ms'] / 1000.0
        line_s = mode['line_ms'] / 1000.0
        cum, t = [], 0.0
        for kind, param, dur in mode['template']:
            cum.append((kind, param, t, dur))
            t += dur

        chans = scan_channels(mode)
        rows = {c: np.zeros((mode['tx_lines'], width)) for c in chans}
        lines_done = 0
        for i in range(mode['tx_lines']):
            line_origin_s = lead_s + i * line_s
            if line_origin_s + line_s > available_s:
                break
            t0_base = line_origin_s * sample_rate
            for kind, param, start_ms, dur_ms in cum:
                if kind != 'scan':
                    continue
                t0 = t0_base + (start_ms / 1000.0) * sample_rate
                dur_samples = (dur_ms / 1000.0) * sample_rate
                rows[param][i] = _sample_row(cs, len(freq), t0, dur_samples, width, 0.0)
            lines_done = i + 1
        if lines_done == 0:
            return None

        if mode['family'] == 'GBR':
            img[:lines_done, :, 0] = rows['R'][:lines_done]
            img[:lines_done, :, 1] = rows['G'][:lines_done]
            img[:lines_done, :, 2] = rows['B'][:lines_done]
            drawn = lines_done
        else:  # PD: each transmission line is two image rows
            img[0:2 * lines_done:2] = _ycrcb_to_rgb(
                rows['Y0'][:lines_done], rows['Cr'][:lines_done], rows['Cb'][:lines_done])
            img[1:2 * lines_done:2] = _ycrcb_to_rgb(
                rows['Y1'][:lines_done], rows['Cr'][:lines_done], rows['Cb'][:lines_done])
            drawn = 2 * lines_done

    elif mode['family'] == 'ROBOT36':
        line_s = mode['line_ms'] / 1000.0
        y_start_ms = mode['sync_ms'] + mode['porch1_ms']
        chroma_start_ms = y_start_ms + mode['y_ms'] + mode['sep_ms'] + mode['porch2_ms']
        y_all = np.zeros((height, width))
        chroma_all = np.zeros((height, width))
        lines_done = 0
        for i in range(height):
            line_origin_s = i * line_s
            if line_origin_s + line_s > available_s:
                break
            t0_base = line_origin_s * sample_rate
            y_all[i] = _sample_row(cs, len(freq), t0_base + (y_start_ms / 1000.0) * sample_rate,
                                    (mode['y_ms'] / 1000.0) * sample_rate, width, 0.0)
            chroma_all[i] = _sample_row(
                cs, len(freq), t0_base + (chroma_start_ms / 1000.0) * sample_rate,
                (mode['chroma_ms'] / 1000.0) * sample_rate, width, 0.0)
            lines_done = i + 1
        if lines_done == 0:
            return None
        # Nominal Cr-then-Cb pairing (matching the encoder's convention) -
        # good enough for a live preview; the real decode confirms this
        # per line from the actual separator tone instead of assuming it.
        for i0 in range(0, lines_done, 2):
            cr_row = chroma_all[i0]
            cb_row = chroma_all[i0 + 1] if i0 + 1 < lines_done else chroma_all[i0]
            img[i0] = _ycrcb_to_rgb(y_all[i0], cr_row, cb_row)
            if i0 + 1 < lines_done:
                img[i0 + 1] = _ycrcb_to_rgb(y_all[i0 + 1], cr_row, cb_row)
        drawn = lines_done
    else:
        return None

    rgb = np.clip(img[:drawn], 0, 255).astype(np.uint8)
    full = np.zeros((height, width, 3), dtype=np.uint8)
    full[:drawn] = rgb
    return Image.fromarray(full, 'RGB'), drawn


class LiveMonitor:
    """
    Owns one microphone stream plus the rolling buffer and waterfall
    derived from it. Call `poll()` on a timer from the GUI thread (a
    few times a second is plenty) to advance detection/decoding;
    everything it updates (`state`, `status_text`, `last_result`,
    `last_error`) is safe to just read from that same thread.
    """
    IDLE, RECEIVING, DECODING, DONE = 'idle', 'receiving', 'decoding', 'done'

    def __init__(self, samplerate=SAMPLE_RATE):
        self.samplerate = samplerate
        self.state = self.IDLE
        self.status_text = "Not listening."
        self.last_result = None    # (PIL.Image, info) once a decode finishes
        self.last_error = None
        self.preview_image = None  # rough, building-up image while RECEIVING

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buf_lock = threading.Lock()
        self._stream = None
        self._last_vis_check_sample = 0
        self._tx_start_sample = None
        self._video_start_sample = None
        self._tx_mode = None
        self._preview_computing = False
        self._last_preview_sample = 0

        self._waterfall = np.zeros((WATERFALL_BINS, WATERFALL_COLS), dtype=np.float32)
        self._wf_lock = threading.Lock()

    # ---------------------------------------------------------------- capture
    def start(self):
        """Opens the default input device. Raises an exception (with a
        readable message) if no microphone is available or accessible -
        the caller should catch this and show it, rather than the app
        just going silently quiet."""
        if self._stream is not None:
            return
        with self._buf_lock:
            self._buffer = np.zeros(0, dtype=np.float32)
        self._last_vis_check_sample = 0
        self._tx_start_sample = None
        self._video_start_sample = None
        self._tx_mode = None
        self.state = self.IDLE
        self.last_result = None
        self.last_error = None
        self.preview_image = None
        self._preview_computing = False
        self._last_preview_sample = 0
        self.status_text = "Listening..."
        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype='float32',
            blocksize=2048, callback=self._audio_callback)
        try:
            self._stream.start()
        except Exception:
            self._stream = None
            raise

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.status_text = "Stopped."

    @property
    def listening(self):
        return self._stream is not None

    def _audio_callback(self, indata, frames, time_info, status):
        # Runs on sounddevice's own audio thread, not the GUI thread.
        chunk = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with self._buf_lock:
            self._buffer = np.concatenate([self._buffer, chunk])
        self._update_waterfall(chunk)

    # -------------------------------------------------------------- waterfall
    def _update_waterfall(self, chunk):
        n = len(chunk)
        if n < 64:
            return
        spec = np.abs(np.fft.rfft(chunk * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.samplerate)
        band = spec[(freqs >= WATERFALL_LO_HZ) & (freqs <= WATERFALL_HI_HZ)]
        if band.size == 0:
            return
        col = np.interp(np.linspace(0, band.size - 1, WATERFALL_BINS),
                         np.arange(band.size), band)
        col = col / (col.max() + 1e-9)
        with self._wf_lock:
            self._waterfall = np.roll(self._waterfall, -1, axis=1)
            self._waterfall[:, -1] = col

    def waterfall_array(self):
        """A (freq_bins, time_cols) array, values ~0-1, oldest column
        first. Purely visual - has no bearing on decode accuracy."""
        with self._wf_lock:
            return self._waterfall.copy()

    # ------------------------------------------------------------ state machine
    def poll(self):
        with self._buf_lock:
            buf = self._buffer  # safe to read outside the lock: we only ever
                                 # rebind self._buffer, never mutate in place
        if self.state == self.IDLE:
            self._check_for_vis(buf)
        elif self.state == self.RECEIVING:
            self._check_reception_progress(buf)
        # DECODING: nothing to do here, the background thread will flip
        # the state to DONE itself once it finishes.

    def _check_for_vis(self, buf):
        n = len(buf)
        step = int(VIS_CHECK_INTERVAL_S * self.samplerate)
        if n - self._last_vis_check_sample < step:
            return
        window = int(VIS_CHECK_WINDOW_S * self.samplerate)
        start = max(0, n - window)
        self._last_vis_check_sample = n
        segment = buf[start:n]
        if len(segment) >= 4096:
            freq = instantaneous_frequency(segment, self.samplerate)
            hdr = find_vis_header(freq, self.samplerate, 0, known_codes=MODES.keys())
            if hdr is not None:
                mode = MODES.get(hdr['vis_code'])
                if mode is not None:
                    self._tx_start_sample = start + hdr['leader_index']
                    # Where line 0 itself actually starts: after the VIS
                    # header (~910ms - two leader tones, break, and 10
                    # data/parity/stop bits) plus a mode's own one-time
                    # leading sync, if it has one (Scottie). Progress and
                    # the live preview are both about the *video*, so
                    # they need to measure from here, not from the
                    # leader's own start - the header's fixed ~910ms
                    # would otherwise silently eat into the video-time
                    # budget on every single transmission.
                    lead_samples = (mode.get('leading_sync_ms', 0.0) / 1000.0) * self.samplerate
                    self._video_start_sample = int(start + hdr['header_end_index'] + lead_samples)
                    self._tx_mode = mode
                    self.state = self.RECEIVING
                    self.status_text = f"Signal detected: {mode['name']} - receiving..."
                    return

        # Keep the idle buffer from growing without bound during a long
        # listening session; nothing before this point is needed anymore.
        cap = int(IDLE_BUFFER_CAP_S * self.samplerate)
        if n > cap:
            trim = n - cap
            with self._buf_lock:
                self._buffer = self._buffer[trim:]
            self._last_vis_check_sample = max(0, self._last_vis_check_sample - trim)

    @staticmethod
    def _expected_duration_s(mode):
        if mode['family'] == 'ROBOT36':
            return (mode['height'] * mode['line_ms']) / 1000.0
        return (mode['leading_sync_ms'] + mode['tx_lines'] * mode['line_ms']) / 1000.0

    def _check_reception_progress(self, buf):
        total = self._expected_duration_s(self._tx_mode)
        elapsed = (len(buf) - self._video_start_sample) / self.samplerate
        pct = max(0, min(100, int(100 * elapsed / total)))
        self.status_text = f"Receiving {self._tx_mode['name']}... {pct}%"

        # Kick off a rough, building-up preview every couple of seconds,
        # in the background so a slow one never blocks polling. Skipped
        # entirely for the trailing edge of the transmission - that's
        # what the real decode below is about to handle properly anyway.
        enough_new_audio = (len(buf) - self._last_preview_sample) >= PREVIEW_MIN_INTERVAL_S * self.samplerate
        if not self._preview_computing and enough_new_audio and elapsed < total - 1.0:
            self._last_preview_sample = len(buf)
            self._preview_computing = True
            snapshot = buf.copy()
            threading.Thread(target=self._compute_preview, args=(snapshot,), daemon=True).start()

        margin_s = 0.5  # a little slack past the nominal duration
        if elapsed < total + margin_s:
            return
        # The segment handed to the real decoder has to start back at the
        # leader (it re-finds the VIS header itself from scratch), but how
        # much of it to take is a video-duration question, so that end
        # bound is measured from _video_start_sample, not _tx_start_sample.
        end = self._video_start_sample + int((total + margin_s) * self.samplerate)
        segment = buf[self._tx_start_sample:min(end, len(buf))].copy()
        self.state = self.DECODING
        self.status_text = "Decoding..."
        threading.Thread(target=self._decode_segment, args=(segment,), daemon=True).start()

    def _compute_preview(self, buf):
        try:
            result = _quick_preview_image(buf, self.samplerate, self._video_start_sample, self._tx_mode)
            if result is not None:
                self.preview_image = result[0]
        except Exception:
            pass  # the preview is a bonus; a hiccup here shouldn't affect the real decode
        finally:
            self._preview_computing = False

    def _decode_segment(self, segment):
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            with wave.open(tmp_path, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.samplerate)
                pcm = np.clip(segment, -1.0, 1.0)
                w.writeframes((pcm * 32767.0).astype(np.int16).tobytes())
            image, info = decode_wav_to_image(tmp_path)
            self.last_result = (image, info)
            self.status_text = f"Decoded: {info['mode']}  ({info['note']})"
        except Exception as e:
            self.last_error = e
            self.status_text = f"Decode failed: {e}"
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            self.state = self.DONE

    def reset_for_next(self):
        """Call after handling a DONE result to go back to watching for
        the next transmission, without restarting the audio stream."""
        self._tx_start_sample = None
        self._video_start_sample = None
        self._tx_mode = None
        self.last_result = None
        self.last_error = None
        self.preview_image = None
        self._preview_computing = False
        self._last_preview_sample = 0
        self.state = self.IDLE
        self.status_text = "Listening..."
