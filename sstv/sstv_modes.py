"""
sstv_modes.py
Timing tables for the supported SSTV modes, keyed by their VIS code.

Constants come from J.L. Barber (N7CXI), "Proposal for SSTV Mode
Specifications" (Dayton SSTV Forum, 2000), cross-checked against the
timing tables used by several independent, currently-maintained SSTV
tools. Every mode below is self-checked at import time: the segment
durations in its `template` are required to add up to its documented
`line_ms` to the nearest hundredth of a millisecond, which catches any
transcription slip immediately rather than producing a subtly-wrong
picture.

Two families cover most color modes and are handled by one generic
engine in the encoder/decoder:

  'GBR' - Martin and Scottie. Each transmission line carries one image
          row as three back-to-back scans: Green, Blue, Red (green
          first, since the eye is most sensitive to it). A sync pulse
          marks each line; Scottie's quirk is that the pulse sits in
          the *middle* of the line (between Blue and Red) rather than
          at the start.

  'PD'  - the PD-family modes. Each transmission line carries *two*
          image rows at once, as Y0 / Cr / Cb / Y1: full luminance for
          the first row, one shared pair of color-difference scans,
          then full luminance for the second row. This is why a PD
          mode's `tx_lines` is half its image height.

Robot 36 is a family of one: it interleaves luminance and (alternating)
chrominance in a way that doesn't fit the same template shape, so it's
described with its own explicit fields and handled by dedicated
encode/decode functions.

A `template` is a list of (kind, param, duration_ms) segments in the
order they're transmitted:
  ('sync', None, ms)   - the 1200Hz line-sync pulse
  ('gap',  None, ms)   - a fixed tone at F_BLACK (a porch/separator)
  ('scan', name, ms)   - one image row's worth of a named channel,
                         spread evenly across `ms`
"""

_MARTIN_GBR_GAP = 0.572
_SCOTTIE_GBR_GAP = 1.5
_PD_PORCH = 2.08


def _martin_template(scan_ms):
    g = _MARTIN_GBR_GAP
    return [
        ('sync', None, 4.862),
        ('gap', None, g),
        ('scan', 'G', scan_ms),
        ('gap', None, g),
        ('scan', 'B', scan_ms),
        ('gap', None, g),
        ('scan', 'R', scan_ms),
        ('gap', None, g),
    ]


def _scottie_template(scan_ms):
    g = _SCOTTIE_GBR_GAP
    return [
        ('gap', None, g),
        ('scan', 'G', scan_ms),
        ('gap', None, g),
        ('scan', 'B', scan_ms),
        ('sync', None, 9.0),
        ('gap', None, g),
        ('scan', 'R', scan_ms),
    ]


def _pd_template(scan_ms):
    return [
        ('sync', None, 20.0),
        ('gap', None, _PD_PORCH),
        ('scan', 'Y0', scan_ms),
        ('scan', 'Cr', scan_ms),
        ('scan', 'Cb', scan_ms),
        ('scan', 'Y1', scan_ms),
    ]


def _cumulative_template(template):
    """[(kind, param, start_ms, duration_ms), ...] from a sequential template."""
    out = []
    t = 0.0
    for kind, param, dur in template:
        out.append((kind, param, t, dur))
        t += dur
    return out


def sync_offset_ms(mode):
    """Where the (one) sync pulse in a GBR/PD template starts, relative
    to that transmission line's own origin. 0.0 for Martin/PD (sync
    leads the line); partway through for Scottie (sync sits mid-line)."""
    for kind, _, start_ms, _ in _cumulative_template(mode['template']):
        if kind == 'sync':
            return start_ms
    raise ValueError("mode template has no sync segment")


def scan_channels(mode):
    """Channel names in a GBR/PD template, in transmission order, deduplicated."""
    seen, out = set(), []
    for kind, param, _, _ in _cumulative_template(mode['template']):
        if kind == 'scan' and param not in seen:
            seen.add(param)
            out.append(param)
    return out


def _make_mode(key, name, vis_code, family, width, height, tx_lines, line_ms,
               template=None, leading_sync_ms=0.0, **extra):
    mode = dict(key=key, name=name, vis_code=vis_code, family=family,
                width=width, height=height, tx_lines=tx_lines, line_ms=line_ms,
                leading_sync_ms=leading_sync_ms)
    if template is not None:
        mode['template'] = template
        total = sum(d for _, _, d in template)
        assert abs(total - line_ms) < 0.01, (
            f"{name}: template sums to {total:.3f}ms, expected {line_ms:.3f}ms")
    mode.update(extra)
    return mode


_MODE_LIST = [
    _make_mode('martin1', 'Martin M1', 44, 'GBR', 320, 256, 256, 446.446,
               template=_martin_template(146.432)),
    _make_mode('martin2', 'Martin M2', 40, 'GBR', 320, 256, 256, 226.798,
               template=_martin_template(73.216)),
    _make_mode('scottie1', 'Scottie S1', 60, 'GBR', 320, 256, 256, 428.220,
               template=_scottie_template(138.240), leading_sync_ms=9.0),
    _make_mode('scottie2', 'Scottie S2', 56, 'GBR', 320, 256, 256, 277.692,
               template=_scottie_template(88.064), leading_sync_ms=9.0),
    _make_mode('scottiedx', 'Scottie DX', 76, 'GBR', 320, 256, 256, 1050.300,
               template=_scottie_template(345.600), leading_sync_ms=9.0),
    _make_mode('pd50', 'PD50', 93, 'PD', 320, 256, 128, 388.160,
               template=_pd_template(91.520)),
    _make_mode('pd90', 'PD90', 99, 'PD', 320, 256, 128, 703.040,
               template=_pd_template(170.240)),
    _make_mode('pd120', 'PD120', 95, 'PD', 640, 496, 248, 508.480,
               template=_pd_template(121.600)),
    # Robot 36: one entry per image line (not per line-pair); see the
    # module docstring. sync+porch1+Y+separator+porch2+chroma = 150.0ms.
    _make_mode('robot36', 'Robot 36', 8, 'ROBOT36', 320, 240, 240, 150.0,
               sync_ms=9.0, porch1_ms=3.0, y_ms=88.0,
               sep_ms=4.5, porch2_ms=1.5, chroma_ms=44.0),
]

for _m in _MODE_LIST:
    if _m['family'] == 'ROBOT36':
        _total = _m['sync_ms'] + _m['porch1_ms'] + _m['y_ms'] + _m['sep_ms'] \
            + _m['porch2_ms'] + _m['chroma_ms']
        assert abs(_total - _m['line_ms']) < 0.01, "Robot 36 timing doesn't add up"

MODES = {m['vis_code']: m for m in _MODE_LIST}
_BY_KEY = {m['key']: m for m in _MODE_LIST}


def get_mode(identifier):
    """Look up a mode by its VIS code (int) or its key/name (str,
    case- and punctuation-insensitive: 'Scottie S1', 'scottie1', and
    'SCOTTIE-1' all resolve to the same mode)."""
    if isinstance(identifier, int):
        if identifier in MODES:
            return MODES[identifier]
        raise KeyError(f"No known mode with VIS code {identifier}")
    key = ''.join(ch for ch in str(identifier).lower() if ch.isalnum())
    for m in _MODE_LIST:
        if m['key'] == key or ''.join(ch for ch in m['name'].lower() if ch.isalnum()) == key:
            return m
    available = ', '.join(m['key'] for m in _MODE_LIST)
    raise KeyError(f"Unknown mode '{identifier}'. Available: {available}")


def list_modes():
    """A short human-readable line per supported mode."""
    lines = []
    for m in _MODE_LIST:
        secs = (m['leading_sync_ms'] + m['tx_lines'] * m['line_ms']) / 1000.0 \
            if m['family'] != 'ROBOT36' else (m['height'] * m['line_ms']) / 1000.0
        lines.append(f"{m['key']:<10} {m['name']:<12} VIS={m['vis_code']:<3} "
                      f"{m['width']}x{m['height']:<4} ~{secs:.1f}s")
    return '\n'.join(lines)
