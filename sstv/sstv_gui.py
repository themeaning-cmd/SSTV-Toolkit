"""
sstv_gui.py
A point-and-click front end for the SSTV encoder/decoder: pick a file,
pick a mode (for encoding), click a button. No command line needed
once this is packaged into an .exe (see the accompanying build
instructions).

Requires sstv_common.py, sstv_modes.py, sstv_encoder.py, and
sstv_decoder.py to be in the same folder - it's a thin GUI layer on
top of that existing library, not a reimplementation.
"""
import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from sstv_modes import MODES
from sstv_encoder import encode_image_to_wav
from sstv_decoder import decode_wav_to_image, scan_for_transmissions

try:
    from PIL import Image, ImageTk, ImageSequence
except ImportError:
    Image = ImageTk = ImageSequence = None

try:
    from sstv_live import LiveMonitor
    LIVE_AVAILABLE = True
    LIVE_IMPORT_ERROR = None
except Exception as e:  # sounddevice missing, no PortAudio, etc.
    LiveMonitor = None
    LIVE_AVAILABLE = False
    LIVE_IMPORT_ERROR = str(e)

# MODES is keyed by VIS code; build a key -> label map (and back) for the
# dropdown, in the same sensible order the modes are defined in.
MODE_LABELS = {m['key']: f"{m['name']}  ({m['width']}x{m['height']})"
               for m in MODES.values()}
LABEL_TO_KEY = {v: k for k, v in MODE_LABELS.items()}


def _run_in_background(fn, on_done):
    """Run fn() on a worker thread; deliver its result (or exception)
    back to the Tk main thread via `.after()`, since Tk widgets can
    only safely be touched from the main thread."""
    def worker():
        try:
            result = fn()
            root.after(0, lambda: on_done(result, None))
        except Exception as e:
            traceback.print_exc()
            root.after(0, lambda: on_done(None, e))
    threading.Thread(target=worker, daemon=True).start()


class BusyBar(ttk.Progressbar):
    """An indeterminate progress bar that only takes space when active,
    so the window doesn't show a static empty bar most of the time.
    Must already be placed once with .grid(...) by the caller (its
    position is remembered across grid_remove()/grid())."""
    def start_busy(self):
        self.grid()
        self.start(12)

    def stop_busy(self):
        self.stop()
        self.grid_remove()


FIT_LABELS = {
    'crop': 'Crop to fill (no borders, edges may be trimmed)',
    'letterbox': 'Letterbox (keeps everything, adds black bars)',
    'stretch': 'Stretch to fill (no borders, may distort proportions)',
}
FIT_LABEL_TO_KEY = {v: k for k, v in FIT_LABELS.items()}


class EncodeTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=14)
        self.image_path = tk.StringVar()

        ttk.Label(self, text="1. Choose an image").grid(row=0, column=0, sticky='w')
        row1 = ttk.Frame(self)
        row1.grid(row=1, column=0, sticky='ew', pady=(2, 12))
        ttk.Entry(row1, textvariable=self.image_path, state='readonly').pack(
            side='left', fill='x', expand=True)
        ttk.Button(row1, text="Browse...", command=self._pick_image).pack(side='left', padx=(6, 0))

        ttk.Label(self, text="2. Choose an SSTV mode").grid(row=2, column=0, sticky='w')
        self.mode_choice = tk.StringVar(value=MODE_LABELS['martin1'])
        mode_box = ttk.Combobox(self, textvariable=self.mode_choice, state='readonly',
                                 values=list(MODE_LABELS.values()), width=40)
        mode_box.grid(row=3, column=0, sticky='ew', pady=(2, 12))

        ttk.Label(self, text="3. If the photo's proportions don't match the mode").grid(
            row=4, column=0, sticky='w')
        self.fit_choice = tk.StringVar(value=FIT_LABELS['crop'])
        fit_box = ttk.Combobox(self, textvariable=self.fit_choice, state='readonly',
                                values=list(FIT_LABELS.values()), width=48)
        fit_box.grid(row=5, column=0, sticky='ew', pady=(2, 12))

        self.encode_btn = ttk.Button(self, text="Encode to WAV...", command=self._encode)
        self.encode_btn.grid(row=6, column=0, sticky='w')

        self.busy = BusyBar(self, mode='indeterminate')
        self.busy.grid(row=7, column=0, sticky='ew', pady=(8, 0))
        self.busy.grid_remove()
        self.status = ttk.Label(self, text="", wraplength=420, foreground='#2a7a2a')
        self.status.grid(row=8, column=0, sticky='w', pady=(10, 0))

        self.columnconfigure(0, weight=1)

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")])
        if path:
            self.image_path.set(path)

    def _encode(self):
        if not self.image_path.get():
            messagebox.showwarning("No image", "Choose an image first.")
            return
        mode_key = LABEL_TO_KEY[self.mode_choice.get()]
        fit_mode = FIT_LABEL_TO_KEY[self.fit_choice.get()]
        default_name = os.path.splitext(os.path.basename(self.image_path.get()))[0] + f"_{mode_key}.wav"
        out_path = filedialog.asksaveasfilename(
            title="Save encoded audio as", defaultextension=".wav",
            initialfile=default_name, filetypes=[("WAV audio", "*.wav")])
        if not out_path:
            return

        self.encode_btn.state(['disabled'])
        self.status.config(text="Encoding...", foreground='#555555')
        self.busy.start_busy()

        def do_encode():
            encode_image_to_wav(self.image_path.get(), mode_key, out_path, fit_mode=fit_mode)
            return out_path

        def done(result, error):
            self.busy.stop_busy()
            self.encode_btn.state(['!disabled'])
            if error:
                self.status.config(text=f"Failed: {error}", foreground='#a12a2a')
                messagebox.showerror("Encoding failed", str(error))
            else:
                self.status.config(text=f"Saved: {result}", foreground='#2a7a2a')

        _run_in_background(do_encode, done)


class DecodeTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=14)
        self.wav_path = tk.StringVar()
        self._last_image = None
        self._photo = None  # keep a reference or Tk garbage-collects the image

        ttk.Label(self, text="1. Choose a WAV recording").grid(row=0, column=0, sticky='w')
        row1 = ttk.Frame(self)
        row1.grid(row=1, column=0, sticky='ew', pady=(2, 12))
        ttk.Entry(row1, textvariable=self.wav_path, state='readonly').pack(
            side='left', fill='x', expand=True)
        ttk.Button(row1, text="Browse...", command=self._pick_wav).pack(side='left', padx=(6, 0))
        ttk.Label(self, text="The mode is detected automatically from the recording -"
                              " you don't need to know it in advance.",
                  wraplength=420, foreground='#666666').grid(row=2, column=0, sticky='w', pady=(0, 12))

        btn_row = ttk.Frame(self)
        btn_row.grid(row=3, column=0, sticky='w')
        self.decode_btn = ttk.Button(btn_row, text="Decode", command=self._decode)
        self.decode_btn.pack(side='left')
        self.scan_btn = ttk.Button(btn_row, text="Scan for multiple transmissions...", command=self._scan)
        self.scan_btn.pack(side='left', padx=(8, 0))

        self.busy = BusyBar(self, mode='indeterminate')
        self.busy.grid(row=4, column=0, sticky='ew', pady=(8, 0))
        self.busy.grid_remove()
        self.status = ttk.Label(self, text="", wraplength=420, foreground='#2a7a2a', justify='left')
        self.status.grid(row=5, column=0, sticky='w', pady=(10, 6))
        self.preview = ttk.Label(self)
        self.preview.grid(row=6, column=0, sticky='w')
        self.save_btn = ttk.Button(self, text="Save image as...", command=self._save_last)
        self.save_btn.grid(row=7, column=0, sticky='w', pady=(8, 0))
        self.save_btn.grid_remove()

        self.columnconfigure(0, weight=1)

    def _pick_wav(self):
        path = filedialog.askopenfilename(
            title="Choose a WAV recording", filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if path:
            self.wav_path.set(path)

    def _set_busy(self, busy, message=""):
        for b in (self.decode_btn, self.scan_btn):
            b.state(['disabled'] if busy else ['!disabled'])
        if busy:
            self.busy.start_busy()
            self.status.config(text=message, foreground='#555555')
        else:
            self.busy.stop_busy()

    def _show_preview(self, image):
        if Image is None:
            return
        thumb = image.copy()
        thumb.thumbnail((360, 280))
        self._photo = ImageTk.PhotoImage(thumb)
        self.preview.configure(image=self._photo)

    def _decode(self):
        if not self.wav_path.get():
            messagebox.showwarning("No file", "Choose a WAV file first.")
            return
        self._set_busy(True, "Decoding (this can take a little while for long recordings)...")

        def do_decode():
            return decode_wav_to_image(self.wav_path.get())

        def done(result, error):
            self._set_busy(False)
            if error:
                self.status.config(text=f"Failed: {error}", foreground='#a12a2a')
                messagebox.showerror("Decoding failed", str(error))
                return
            image, info = result
            self._last_image = image
            self._show_preview(image)
            self.status.config(
                text=(f"Detected mode: {info['mode']}  (VIS parity OK: {info['parity_ok']})\n"
                      f"Frequency offset: {info['frequency_offset_hz']:+.1f} Hz\n"
                      f"{info['note']}"),
                foreground='#2a7a2a')
            self.save_btn.grid()

        _run_in_background(do_decode, done)

    def _save_last(self):
        if self._last_image is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save decoded image as", defaultextension=".png",
            filetypes=[("PNG image", "*.png")])
        if path:
            self._last_image.save(path)
            self.status.config(text=f"Saved: {path}", foreground='#2a7a2a')

    def _scan(self):
        if not self.wav_path.get():
            messagebox.showwarning("No file", "Choose a WAV file first.")
            return
        out_dir = filedialog.askdirectory(title="Choose a folder to save any decoded images into")
        if not out_dir:
            return
        self._set_busy(True, "Scanning the recording for transmissions...")

        def do_scan():
            results = scan_for_transmissions(self.wav_path.get(), verbose=False)
            saved = []
            for i, tx in enumerate(results):
                if tx.get('image') is not None:
                    fname = f"transmission_{i+1}_{tx['mode'].replace(' ', '')}_{int(tx['time_s'])}s.png"
                    fpath = os.path.join(out_dir, fname)
                    tx['image'].save(fpath)
                    saved.append((tx['time_s'], tx['mode'], fpath))
            return saved

        def done(result, error):
            self._set_busy(False)
            if error:
                self.status.config(text=f"Failed: {error}", foreground='#a12a2a')
                messagebox.showerror("Scan failed", str(error))
                return
            if not result:
                self.status.config(text="No SSTV transmissions found in this recording.",
                                    foreground='#555555')
                return
            lines = [f"{t:.1f}s - {m}" for t, m, _ in result]
            self.status.config(
                text=f"Found {len(result)} transmission(s), saved to {out_dir}:\n" + "\n".join(lines),
                foreground='#2a7a2a')

        _run_in_background(do_scan, done)


def _waterfall_to_image(arr):
    """(freq_bins, time_cols) float array, ~0-1 -> a colored PIL image.
    Purely a visual convenience, has no bearing on decode accuracy."""
    v = np.clip(arr, 0.0, 1.0)
    stops = [0.0, 0.35, 0.7, 1.0]
    r = np.interp(v, stops, [0, 10, 40, 255])
    g = np.interp(v, stops, [0, 20, 180, 255])
    b = np.interp(v, stops, [15, 90, 200, 255])
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb = np.flipud(rgb)  # higher pitch renders toward the top
    return Image.fromarray(rgb, 'RGB')


class LiveTab(ttk.Frame):
    """Continuously listens on the microphone, shows a scrolling
    waterfall of what's coming in, and decodes automatically the
    moment an SSTV VIS header is recognized. Deliberately does not
    smooth over what a casual mic capture actually picks up - the
    image that comes out reflects real room noise, echo, and whatever
    clock wobble the sound card has, same as sstv_decoder.py always
    has for any recording."""

    def __init__(self, master):
        super().__init__(master, padding=14)
        self.monitor = LiveMonitor() if LIVE_AVAILABLE else None
        self._last_image = None
        self._photo = None
        self._wf_photo = None
        self._shown_preview = None

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky='ew')
        self.toggle_btn = ttk.Button(top, text="Start Listening", command=self._toggle)
        self.toggle_btn.pack(side='left')
        self.next_btn = ttk.Button(top, text="Listen for next", command=self._reset)
        self.next_btn.pack(side='left', padx=(8, 0))
        self.next_btn.pack_forget()

        self.wf_label = ttk.Label(self)
        self.wf_label.grid(row=1, column=0, pady=(10, 6))
        self._set_waterfall_placeholder()

        self.status = ttk.Label(self, text="Not listening.", wraplength=420,
                                 foreground='#555555', justify='left')
        self.status.grid(row=2, column=0, sticky='w')

        self.preview = ttk.Label(self)
        self.preview.grid(row=3, column=0, pady=(10, 0))
        self.save_btn = ttk.Button(self, text="Save image as...", command=self._save_last)
        self.save_btn.grid(row=4, column=0, sticky='w', pady=(8, 0))
        self.save_btn.grid_remove()

        if not LIVE_AVAILABLE:
            self.status.config(
                text=("Live monitoring isn't available: " + LIVE_IMPORT_ERROR +
                      "\nInstall it with:  py -m pip install sounddevice"),
                foreground='#a12a2a')
            self.toggle_btn.state(['disabled'])

        self.columnconfigure(0, weight=1)
        self._tick()

    def _set_waterfall_placeholder(self):
        blank = Image.new('RGB', (440, 180), (10, 12, 18))
        self._wf_photo = ImageTk.PhotoImage(blank)
        self.wf_label.configure(image=self._wf_photo)

    def _toggle(self):
        if self.monitor.listening:
            self.monitor.stop()
            self.toggle_btn.config(text="Start Listening")
        else:
            try:
                self.monitor.start()
                self.toggle_btn.config(text="Stop Listening")
                self.next_btn.pack_forget()
                self.preview.configure(image='')
                self.save_btn.grid_remove()
                self._last_image = None
                self._shown_preview = None
            except Exception as e:
                messagebox.showerror(
                    "Couldn't open microphone",
                    f"{e}\n\nCheck that a microphone is connected and that Windows' "
                    "microphone privacy setting allows desktop apps to use it.")

    def _reset(self):
        self.monitor.reset_for_next()
        self.next_btn.pack_forget()
        self.preview.configure(image='')
        self.save_btn.grid_remove()
        self._last_image = None
        self._shown_preview = None

    def _save_last(self):
        if self._last_image is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save decoded image as", defaultextension=".png",
            filetypes=[("PNG image", "*.png")])
        if path:
            self._last_image.save(path)
            self.status.config(text=f"Saved: {path}", foreground='#2a7a2a')

    def _tick(self):
        if LIVE_AVAILABLE and self.monitor.listening:
            self.monitor.poll()
            wf_img = _waterfall_to_image(self.monitor.waterfall_array())
            wf_img = wf_img.resize((440, 180))
            self._wf_photo = ImageTk.PhotoImage(wf_img)
            self.wf_label.configure(image=self._wf_photo)

            color = '#555555'
            if self.monitor.state == self.monitor.RECEIVING:
                color = '#8a6d1a'
            elif self.monitor.state == self.monitor.DONE:
                color = '#a12a2a' if self.monitor.last_error else '#2a7a2a'
            self.status.config(text=self.monitor.status_text, foreground=color)

            if self.monitor.state == self.monitor.RECEIVING \
                    and self.monitor.preview_image is not self._shown_preview:
                self._shown_preview = self.monitor.preview_image
                thumb = self.monitor.preview_image.copy()
                thumb.thumbnail((360, 280))
                self._photo = ImageTk.PhotoImage(thumb)
                self.preview.configure(image=self._photo)

            if self.monitor.state == self.monitor.DONE and self._last_image is None \
                    and self.monitor.last_result is not None:
                image, info = self.monitor.last_result
                self._last_image = image
                thumb = image.copy()
                thumb.thumbnail((360, 280))
                self._photo = ImageTk.PhotoImage(thumb)
                self.preview.configure(image=self._photo)
                self.save_btn.grid()
                self.next_btn.pack(side='left', padx=(8, 0))
                self.status.config(
                    text=(f"Decoded: {info['mode']}  (VIS parity OK: {info['parity_ok']})\n"
                          f"Frequency offset: {info['frequency_offset_hz']:+.1f} Hz\n"
                          f"{info['note']}"),
                    foreground='#2a7a2a')

        self.after(150, self._tick)


def _app_dir():
    """Where to look for optional user-supplied files (like a mascot
    photo) next to the app - the exe's own folder when running as a
    packaged executable, or this script's folder otherwise. Not the
    same as the PyInstaller bundle's internal temp dir, since a file
    the user drops in next to the exe was never bundled into it."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _load_logo_frames(max_height=42):
    """An animated GIF logo for the header, entirely optional: drop a
    file named logo.gif next to this app and it plays in place of the
    plain text title, scaled down to max_height tall (aspect ratio
    preserved, every frame resized the same so playback doesn't jitter
    in size). Returns (photo_frames, durations_ms), or (None, None) if
    absent/unreadable, in which case the caller should fall back to
    plain text - never left half-drawn either way."""
    if Image is None or ImageSequence is None:
        return None, None
    path = os.path.join(_app_dir(), 'logo.gif')
    if not os.path.exists(path):
        return None, None
    try:
        src = Image.open(path)
        frames, durations = [], []
        for frame in ImageSequence.Iterator(src):
            rgba = frame.convert('RGBA')
            if rgba.height > max_height:
                scale = max_height / rgba.height
                new_size = (max(1, round(rgba.width * scale)), max_height)
                rgba = rgba.resize(new_size, Image.Resampling.LANCZOS)
            frames.append(ImageTk.PhotoImage(rgba))
            durations.append(max(20, frame.info.get('duration', 100)))  # guard against a 0ms quirk
        print(f"[logo] Loaded {path}: {len(frames)} frame(s), "
              f"{frames[0].width()}x{frames[0].height()} after scaling")
        return frames, durations
    except Exception as e:
        print(f"[logo] Found {path} but couldn't load it: {type(e).__name__}: {e}")
        return None, None


def _load_mascot_photo(max_size=84):
    """An optional little corner photo, entirely user-supplied: drop an
    image file starting with "mascot" (mascot.png, mascot_png.png,
    mascot-photo.jpg, ...) next to this app and it shows up
    automatically - matching loosely on purpose, since "save image as"
    dialogs often mangle the suggested name. Absent or unreadable ->
    None. Prints a plain-English reason to the console either way (the
    .bat launcher keeps a console window open specifically so this
    kind of thing is visible instead of failing invisibly)."""
    if Image is None:
        print("[mascot] Pillow isn't available, so no image can be shown.")
        return None
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    try:
        candidates = sorted(
            f for f in os.listdir(_app_dir())
            if f.lower().startswith('mascot') and f.lower().endswith(exts))
    except OSError as e:
        print(f"[mascot] Couldn't list {_app_dir()}: {e}")
        return None
    if not candidates:
        print(f"[mascot] No file starting with 'mascot' found in {_app_dir()}")
        return None
    path = os.path.join(_app_dir(), candidates[0])
    try:
        size_kb = os.path.getsize(path) / 1024
        img = Image.open(path)
        fmt, mode = img.format, img.mode
        # Keep real transparency where the file actually has it (a
        # background-removed PNG, typically) instead of flattening it
        # onto a solid color - that's what would show up as a visible
        # white/black box around the subject instead of a clean overlay.
        has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
        img = img.convert('RGBA') if has_alpha else img.convert('RGB')
        img.thumbnail((max_size, max_size))
        print(f"[mascot] Loaded {path} ({size_kb:.0f} KB, format={fmt}, "
              f"mode={mode}, transparent={has_alpha}) -> thumbnail {img.size}")
        return img
    except Exception as e:
        print(f"[mascot] Found {path} but couldn't load it: {type(e).__name__}: {e}")
        return None


def main():
    global root
    root = tk.Tk()
    root.title("SSTV Toolkit")
    root.geometry("500x760")
    root.resizable(False, False)  # also disables maximize - Tk ties the two together

    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')

    bg = '#aae5a4'
    root.configure(background=bg)
    style.configure('.', background=bg)
    style.configure('TFrame', background=bg)
    style.configure('TLabel', background=bg)
    style.configure('TButton', background=bg)
    style.configure('TCheckbutton', background=bg)
    style.configure('TRadiobutton', background=bg)
    style.configure('TNotebook', background=bg)
    style.configure('TNotebook.Tab', background=bg)
    style.map('TNotebook.Tab', background=[('selected', bg)])
    # Entry/Combobox/Progressbar deliberately left at their normal
    # (usually white) appearance - a green text-entry field would be
    # harder to read and blur the line between editable and decorative.

    header = ttk.Frame(root, padding=(14, 12, 14, 0))
    header.pack(fill='x')
    logo_frames, logo_durations = _load_logo_frames()
    if logo_frames:
        logo_label = ttk.Label(header, image=logo_frames[0])
        logo_label.image = logo_frames[0]
        logo_label.pack(anchor='w')

        def _animate_logo(idx=0):
            logo_label.configure(image=logo_frames[idx])
            logo_label.image = logo_frames[idx]  # keep a reference each frame too
            root.after(logo_durations[idx], _animate_logo, (idx + 1) % len(logo_frames))

        _animate_logo()
    else:
        ttk.Label(header, text="SSTV Toolkit", font=('Segoe UI', 14, 'bold')).pack(anchor='w')

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=8, pady=8)
    notebook.add(EncodeTab(notebook), text="Encode")
    notebook.add(DecodeTab(notebook), text="Decode")
    live_tab = LiveTab(notebook)
    notebook.add(live_tab, text="Live Monitor")

    # Placed (not packed) directly on root, after the notebook, so it
    # floats in the corner on top of whatever tab is showing rather
    # than reserving its own strip of the window.
    mascot = _load_mascot_photo()
    if mascot is not None:
        mascot_photo = ImageTk.PhotoImage(mascot)
        mascot_label = ttk.Label(root, image=mascot_photo)
        mascot_label.image = mascot_photo  # keep a reference - Tk drops the image without one
        mascot_label.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor='se')

    def on_close():
        if LIVE_AVAILABLE and live_tab.monitor.listening:
            live_tab.monitor.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


if __name__ == '__main__':
    main()
