"""
example_usage.py
Runnable walkthrough of the encoder and decoder. With no arguments it
generates its own test image so you can try the whole pipeline with
nothing but `python3 example_usage.py`.

    python3 example_usage.py                    # self-contained demo
    python3 example_usage.py my_photo.jpg        # encode/decode your own image
    python3 example_usage.py my_photo.jpg pd120  # ...in a specific mode
"""
import sys

from sstv_modes import list_modes
from sstv_encoder import encode_image_to_wav
from sstv_decoder import decode_wav_to_image, scan_for_transmissions


def _make_test_image(path):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (640, 480), (25, 25, 35))
    d = ImageDraw.Draw(img)
    for i, c in enumerate([(220, 40, 40), (40, 180, 60), (50, 90, 220), (230, 210, 40)]):
        d.rectangle([i * 160, 0, i * 160 + 160, 240], fill=c)
    for x in range(640):
        v = int(255 * x / 640)
        d.line([(x, 250), (x, 400)], fill=(v, 255 - v, 160))
    d.text((20, 420), 'SSTV EXAMPLE', fill=(255, 255, 255))
    img.save(path)
    return path


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else _make_test_image('demo_source.png')
    mode_name = sys.argv[2] if len(sys.argv) > 2 else 'martin1'

    print("Supported modes:")
    print(list_modes())
    print()

    # --- Encode: image -> audio -----------------------------------------
    wav_path = f'demo_{mode_name}.wav'
    print(f"Encoding {image_path!r} as {mode_name}...")
    encode_image_to_wav(image_path, mode_name, wav_path)
    print(f"  wrote {wav_path} (play this into a radio, or straight into an SDR's TX audio)")

    # --- Decode: audio -> image, mode auto-detected from the VIS header -
    png_path = f'demo_{mode_name}_decoded.png'
    print(f"\nDecoding {wav_path!r} (mode NOT told to the decoder - it reads the VIS header)...")
    image, info = decode_wav_to_image(wav_path, png_path)
    print(f"  detected mode : {info['mode']} (VIS {info['vis_code']}, parity_ok={info['parity_ok']})")
    print(f"  freq. offset  : {info['frequency_offset_hz']:+.1f} Hz")
    print(f"  line sync     : {info['note']}")
    print(f"  wrote {png_path}")

    # --- Scan a longer recording for every transmission in it -----------
    print("\nscan_for_transmissions() finds every SSTV transmission in a long "
          "recording on its own - no need to know in advance how many there "
          "are, where they start, or what mode each one uses. Pass it any "
          "WAV file (e.g. a monitoring capture) and it reports each one it finds:")
    print(f"  results = scan_for_transmissions({wav_path!r})")
    results = scan_for_transmissions(wav_path)
    print(f"  -> found {len(results)} transmission(s) in this file")


if __name__ == '__main__':
    main()
