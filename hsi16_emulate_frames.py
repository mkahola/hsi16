#!/usr/bin/env python3
"""
hsi16_emulate_frames.py

Data-plane emulator for the HSI16 16-band mosaic sensor.

This does NOT touch the kernel hsi16.c driver or CSI-2 at all -- it
feeds synthetic frames into a v4l2loopback device so the userspace
demosaic / radiometric-calibration code can be developed and unit
tested against known ground truth, decoupled from whatever kernel
driver (real or fake-hw) is under test.

Two output formats, selected with --format:

  y16   (default) -- V4L2_PIX_FMT_Y16, 16-bit grey, values 0-1023
          (10-bit range in a 16-bit container, matching the sensor's
          native ADC width). This is the format to feed a real
          demosaic/calibration pipeline against: full precision, no
          information thrown away.

  grey  -- V4L2_PIX_FMT_GREY, 8-bit grey, values 0-255 (top 8 bits
          of the same 10-bit value, i.e. right-shifted by 2). This
          format exists purely so the pattern can be eyeballed:
          V4L2_PIX_FMT_GREY / VLC's "GREY" rawvideo chroma are a
          well-established 8-bit grayscale pairing, unlike a 16-bit
          grey chroma name in VLC, which isn't something I could
          verify exists -- so GREY is the guaranteed-viewable path.
          Do not use --format grey for anything other than visual
          sanity-checking; the low 2 bits are gone.

Setup:
    sudo modprobe v4l2loopback video_nr=10 card_label="HSI16 Emulator" \
        exclusive_caps=1
    python3 hsi16_emulate_frames.py --device /dev/video10 --format y16

To view the pattern in VLC (8-bit mode only -- see --format above):
    python3 hsi16_emulate_frames.py --device /dev/video10 --format grey \
        --width 512 --height 384
    vlc --demux rawvideo --rawvid-width 512 --rawvid-height 384 \
        --rawvid-fps 5 --rawvid-chroma=GREY /dev/video10

(Smaller width/height suggested for the VLC path purely so the
mosaic cells are large enough on screen to see the 4x4 structure
clearly -- the emulator works at any size that's a multiple of 4.)

Each pixel's value encodes (band_index * 64 + spatial_ramp) so a
demosaic implementation under test can be checked both for "did it
route each mosaic position to the correct band" (upper bits) and
"did it preserve spatial detail within a band" (lower bits, a simple
ramp/gradient per band). In --format grey, expect to see a repeating
4x4 grid of cells, each one 16 visibly distinct brightness steps
apart with a faint gradient inside -- not a photo, a synthetic test
pattern.
"""

import argparse
import fcntl
import struct
import sys
import time

import numpy as np

# Minimal v4l2 ioctl constants (avoids a hard dependency on
# python-v4l2 / v4l2py for this small a footprint).
VIDIOC_S_FMT = 0xc0d05605
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1

MOSAIC_PERIOD = 4          # 4x4 = 16 bands, matches hsi16.c
BAND_COUNT = MOSAIC_PERIOD * MOSAIC_PERIOD
ADC_MAX = 1023              # 10-bit

FORMATS = {
    "y16": {
        "fourcc": b"Y16 ",
        "bytes_per_px": 2,
        "dtype": np.uint16,
    },
    "grey": {
        "fourcc": b"GREY",
        "bytes_per_px": 1,
        "dtype": np.uint8,
    },
}


def build_mosaic_frame(width: int, height: int, fmt: str) -> np.ndarray:
    """Synthetic 16-band mosaic frame.

    band_index = (row % 4) * 4 + (col % 4)   -- matches a simple
    row-major 4x4 super-pixel layout; adjust to your real filter
    layout table before trusting downstream results against it.
    """
    rows = np.arange(height).reshape(-1, 1) % MOSAIC_PERIOD
    cols = np.arange(width).reshape(1, -1) % MOSAIC_PERIOD
    band_index = (rows * MOSAIC_PERIOD + cols).astype(np.uint16)

    # Low-frequency spatial ramp per band so within-band detail is
    # also checkable, not just band routing.
    ramp = (
        (np.linspace(0, 63, width, dtype=np.uint16).reshape(1, -1))
        + (np.linspace(0, 63, height, dtype=np.uint16).reshape(-1, 1))
    ) % 64

    frame_10bit = (band_index * 64 + ramp).astype(np.uint16)
    np.clip(frame_10bit, 0, ADC_MAX, out=frame_10bit)

    if fmt == "y16":
        return frame_10bit
    elif fmt == "grey":
        # Top 8 bits of the 10-bit value -- lossy, view-only.
        return (frame_10bit >> 2).astype(np.uint8)
    else:
        raise ValueError(f"unknown format {fmt!r}")


def set_output_format(fd: int, width: int, height: int, fmt: str) -> None:
    spec = FORMATS[fmt]
    bpp = spec["bytes_per_px"]
    pixelformat = struct.unpack("<I", spec["fourcc"])[0]

    # struct v4l2_format for V4L2_BUF_TYPE_VIDEO_OUTPUT, pix member:
    # type(u32) + width(u32) height(u32) pixelformat(u32) field(u32)
    # + bytesperline(u32) + sizeimage(u32) + padding. Packed manually
    # to avoid pulling in ctypes struct defs.
    fmt_struct = struct.pack(
        "<IIIIIIIIII200x",
        V4L2_BUF_TYPE_VIDEO_OUTPUT,
        width,
        height,
        pixelformat,
        V4L2_FIELD_NONE,
        width * bpp,          # bytesperline
        width * height * bpp, # sizeimage
        0, 0, 0,
    )
    fcntl.ioctl(fd, VIDIOC_S_FMT, fmt_struct)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="/dev/video10",
                     help="v4l2loopback output device node")
    ap.add_argument("--format", choices=sorted(FORMATS), default="y16",
                     help="y16 = full precision (default, for real testing); "
                          "grey = 8-bit, guaranteed VLC-viewable, view-only")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1536)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--count", type=int, default=0,
                     help="frames to emit, 0 = run forever")
    args = ap.parse_args()

    if args.width % MOSAIC_PERIOD or args.height % MOSAIC_PERIOD:
        print(f"width/height must be multiples of {MOSAIC_PERIOD}",
              file=sys.stderr)
        return 1

    frame = build_mosaic_frame(args.width, args.height, args.format)
    payload = frame.tobytes()

    with open(args.device, "wb", buffering=0) as f:
        set_output_format(f.fileno(), args.width, args.height, args.format)

        period = 1.0 / args.fps
        n = 0
        while args.count == 0 or n < args.count:
            f.write(payload)
            n += 1
            time.sleep(period)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
