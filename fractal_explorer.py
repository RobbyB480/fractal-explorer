#!/usr/bin/env python3
"""
fractal_explorer.py — CRT Fractal Explorer v3
blessed TUI for live navigation + iTerm2 high-res blast.

Keys:
  arrows             — pan
  z / x             — zoom in / out (centered)
  m                 — cycle fractal mode (Mandelbrot → Burning Ship → Tricorn)
  p                 — cycle palette
  j                 — toggle Julia mode
  o                 — trace orbit at viewport center
  c                 — clear orbit
  [ / ]             — iterations down / up
  1-6               — presets
  0                 — reset view
  r                 — blast full iTerm2 high-res render
  S                 — save PNG to Desktop
  q                 — quit
"""
from __future__ import annotations
import sys, io, base64, time, pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import blessed

# ── Constants ─────────────────────────────────────────────────────────────────
FRACTAL_MODES  = ["mandelbrot", "burning_ship", "tricorn"]
FRACTAL_LABELS = {"mandelbrot":"MANDELBROT","burning_ship":"BURNING SHIP","tricorn":"TRICORN"}
FRACTAL_EMOJI  = {"mandelbrot":"🦞","burning_ship":"🔥","tricorn":"🦄"}
DEFAULT_COORDS = {
    "mandelbrot":   (-2.5, 1.0,-1.25, 1.25),
    "burning_ship": (-2.5, 1.5, -0.2, 1.8),
    "tricorn":      (-2.5, 1.0,-1.25, 1.25),
}
PRESETS = {
    "1":("SEAHORSE VALLEY", -0.7462,-0.7430, 0.1095, 0.1127),
    "2":("ELEPHANT VALLEY",  0.2549, 0.2581,-0.0006, 0.0026),
    "3":("DOUBLE SPIRAL",   -0.7269,-0.7249, 0.1889, 0.1909),
    "4":("TENDRILS",        -1.786, -1.781, -0.002,  0.003 ),
    "5":("MINI BROT",       -1.768, -1.764, -0.002,  0.002 ),
    "6":("LIGHTNING",       -0.5015,-0.4985, 0.5625, 0.5655),
    "7":("MISIUREWICZ ∞",  -0.10109636384562-2e-14, -0.10109636384562+2e-14,
                              0.95628651080914-2e-14,  0.95628651080914+2e-14),
    "8":("NEEDLE ∞",         -1.9999219876-1e-4, -1.9999019876+1e-4,
                              -1e-4,                   1e-4),
    "9":("SEAHORSE ∞",      -0.7436447860-3e-8, -0.7436447860+3e-8,
                               0.1318252536-3e-8,  0.1318252536+3e-8),
    "0":("ANTENNA ∞",        -1.5554567-3e-8,    -1.5554567+3e-8,
                              -3e-8,               3e-8),
}
PALETTES = ["hybrid","red_crt","amber","fire","ice","ultra","psychedelic","grayscale"]
PALETTE_LABELS = {
    "hybrid":"HYBRID","red_crt":"RED CRT","amber":"AMBER","fire":"FIRE",
    "ice":"ICE","ultra":"ULTRA","psychedelic":"PSYCH","grayscale":"GRAY",
}

# ── Palettes ──────────────────────────────────────────────────────────────────
def apply_palette(name, t):
    t = np.clip(t, 0.0, 1.0); sh = t.shape
    if name == "red_crt":
        v = np.power(t, 0.45)
        return np.stack([np.clip(v*255,0,255).astype(np.uint8),
                         np.clip(v*15, 0,255).astype(np.uint8),
                         np.clip(v*15, 0,255).astype(np.uint8)], axis=-1)
    if name == "amber":
        v = np.power(t, 0.45)
        return np.stack([np.clip(v*255,0,255).astype(np.uint8),
                         np.clip(v*175,0,255).astype(np.uint8),
                         np.zeros(sh,dtype=np.uint8)], axis=-1)
    if name == "hybrid":
        rgb = np.zeros((*sh,3),dtype=np.uint8); lo = t < 0.5
        v = np.power(t[lo]*2,0.6)
        rgb[lo,0]=np.clip(v*255,0,255).astype(np.uint8)
        rgb[lo,1]=np.clip(v*60, 0,255).astype(np.uint8)
        hi = ~lo; v = np.power((t[hi]-0.5)*2,0.6)
        rgb[hi,0]=np.clip(v*20, 0,255).astype(np.uint8)
        rgb[hi,1]=np.clip(v*255,0,255).astype(np.uint8)
        rgb[hi,2]=np.clip(v*100,0,255).astype(np.uint8)
        return rgb
    if name == "fire":
        return np.stack([
            np.clip(np.power(t*1.5,0.7)*255,0,255).astype(np.uint8),
            np.clip(np.power(np.maximum(0,t*2-0.5),1.2)*255,0,255).astype(np.uint8),
            np.zeros(sh,dtype=np.uint8)], axis=-1)
    if name == "ice":
        v = np.power(t,0.5)
        return np.stack([np.clip(v*20, 0,255).astype(np.uint8),
                         np.clip(v*120,0,255).astype(np.uint8),
                         np.clip(v*255,0,255).astype(np.uint8)], axis=-1)
    if name == "ultra":
        c = t*np.pi*6
        return np.stack([(128+127*np.sin(c)).astype(np.uint8),
                         (128+127*np.sin(c+2.094)).astype(np.uint8),
                         (128+127*np.sin(c+4.189)).astype(np.uint8)], axis=-1)
    if name == "psychedelic":
        c = t*np.pi*10
        return np.stack([(128+127*np.sin(c*1.3)).astype(np.uint8),
                         (128+127*np.sin(c*0.7+1)).astype(np.uint8),
                         (128+127*np.cos(c*1.9+2)).astype(np.uint8)], axis=-1)
    v = np.clip(np.power(t,0.5)*255,0,255).astype(np.uint8)
    return np.stack([v,v,v], axis=-1)

# ── Fractal computation — engine priority: MLX (M4 GPU) > numba > numpy ───────
import math as _math
import pathlib as _pathlib

# ── Try MLX (Apple Silicon GPU via unified memory) ────────────────────────────
try:
    import mlx.core as _mx
    _MLX = True
except ImportError:
    _MLX = False

# ── Try numba (CPU JIT, parallel) ─────────────────────────────────────────────
try:
    from numba import njit as _njit, prange as _prange
    _NUMBA = True
except ImportError:
    _NUMBA = False

# ── Try mpmath (arbitrary precision — needed for perturbation reference orbit) ─
try:
    import mpmath as _mp
    _MPMATH = True
except Exception as _mpmath_err:
    _MPMATH = False
    import pathlib, traceback
    _elog = pathlib.Path.home() / "fractal_perturb.log"
    with open(_elog, "a") as _ef:
        _ef.write(f"[MPMATH-IMPORT-ERROR] {type(_mpmath_err).__name__}: {_mpmath_err}\n")
        _ef.write(traceback.format_exc() + "\n")

# ── Perturbation zoom threshold ────────────────────────────────────────────────
_PERTURB_ZOOM_LIMIT = 1e13

# ── Engine label for UI ───────────────────────────────────────────────────────
if _MLX and _MPMATH:
    _ENGINE = "MLX ⚡ M4 · numba · perturbation ∞"
elif _MLX:
    _ENGINE = "MLX GPU ⚡ M4 + numba CPU fallback"
elif _NUMBA and _MPMATH:
    _ENGINE = "numba JIT ⚡ CPU · perturbation ∞"
elif _NUMBA:
    _ENGINE = "numba JIT ⚡ CPU"
else:
    _ENGINE = "numpy CPU"

# ── numba JIT functions (defined unconditionally when numba available) ─────────
if _NUMBA:
    @_njit(parallel=True, cache=True, fastmath=True)
    def _jit_mandelbrot(xmin,xmax,ymin,ymax,W,H,max_iter):
        out=np.zeros((H,W),dtype=np.float64)
        for row in _prange(H):
            ci=ymin+(row/H)*(ymax-ymin)
            for col in range(W):
                cr=xmin+(col/W)*(xmax-xmin)
                zr=0.0; zi=0.0; i=0
                while i<max_iter:
                    zr2=zr*zr; zi2=zi*zi
                    if zr2+zi2>4.0:
                        log_zn=_math.log(zr2+zi2)/2.0
                        nu=_math.log(max(log_zn/_math.log(2.0),1e-10))/_math.log(2.0)
                        out[row,col]=max(0.0,i+1.0-nu); break
                    zi=2.0*zr*zi+ci; zr=zr2-zi2+cr; i+=1
        return out

    @_njit(parallel=True, cache=True, fastmath=True)
    def _jit_burning_ship(xmin,xmax,ymin,ymax,W,H,max_iter):
        out=np.zeros((H,W),dtype=np.float64)
        for row in _prange(H):
            ci=ymin+(row/H)*(ymax-ymin)
            for col in range(W):
                cr=xmin+(col/W)*(xmax-xmin)
                zr=0.0; zi=0.0; i=0
                while i<max_iter:
                    zr2=zr*zr; zi2=zi*zi
                    if zr2+zi2>4.0:
                        log_zn=_math.log(zr2+zi2)/2.0
                        nu=_math.log(max(log_zn/_math.log(2.0),1e-10))/_math.log(2.0)
                        out[row,col]=max(0.0,i+1.0-nu); break
                    zi=2.0*abs(zr)*abs(zi)+ci; zr=zr2-zi2+cr; i+=1
        return out

    @_njit(parallel=True, cache=True, fastmath=True)
    def _jit_tricorn(xmin,xmax,ymin,ymax,W,H,max_iter):
        out=np.zeros((H,W),dtype=np.float64)
        for row in _prange(H):
            ci=ymin+(row/H)*(ymax-ymin)
            for col in range(W):
                cr=xmin+(col/W)*(xmax-xmin)
                zr=0.0; zi=0.0; i=0
                while i<max_iter:
                    zr2=zr*zr; zi2=zi*zi
                    if zr2+zi2>4.0:
                        log_zn=_math.log(zr2+zi2)/2.0
                        nu=_math.log(max(log_zn/_math.log(2.0),1e-10))/_math.log(2.0)
                        out[row,col]=max(0.0,i+1.0-nu); break
                    zi=-2.0*zr*zi+ci; zr=zr2-zi2+cr; i+=1
        return out

    @_njit(parallel=True, cache=True, fastmath=True)
    def _jit_julia(xmin,xmax,ymin,ymax,W,H,max_iter,jcr,jci):
        out=np.zeros((H,W),dtype=np.float64)
        for row in _prange(H):
            zi=ymin+(row/H)*(ymax-ymin)
            for col in range(W):
                zr=xmin+(col/W)*(xmax-xmin); i=0
                while i<max_iter:
                    zr2=zr*zr; zi2=zi*zi
                    if zr2+zi2>4.0:
                        log_zn=_math.log(zr2+zi2)/2.0
                        nu=_math.log(max(log_zn/_math.log(2.0),1e-10))/_math.log(2.0)
                        out[row,col]=max(0.0,i+1.0-nu); break
                    nzi=2.0*zr*zi+jci; zr=zr2-zi2+jcr; zi=nzi; i+=1
        return out



# ── Perturbation Theory Engine ────────────────────────────────────────────────
# At extreme zoom the viewport is so narrow that float64 pixel coordinates
# all round to the same value — adjacent pixels become indistinguishable.
# Fix: compute ALL coordinates (reference orbit + delta arrays) in mpmath
# Three rules for correct deep zoom perturbation:
#   1. Reference orbit computed ONCE in mpmath at 200+ bits before any loop
#   2. Pixel size = 3.5 / (zoom * W) — one tiny number, never from coord subtraction
#   3. Z_ref + δZ only computed at escape check / smooth colouring — not inside loop

def _reference_orbit(cr_mp, ci_mp, max_iter):
    """Compute Mandelbrot reference orbit at (cr_mp, ci_mp) in full mpmath precision.
    Called ONCE before pixel loop. Returns float64 arrays for numba JIT."""
    zr = _mp.mpf(0); zi = _mp.mpf(0)
    orbit_r = []; orbit_i = []
    for _ in range(max_iter):
        orbit_r.append(float(zr))
        orbit_i.append(float(zi))
        zr2 = zr * zr;  zi2 = zi * zi
        if zr2 + zi2 > 4:
            break
        new_zi = 2 * zr * zi + ci_mp
        zr     = zr2 - zi2 + cr_mp
        zi     = new_zi
    return (np.array(orbit_r, dtype=np.float64),
            np.array(orbit_i, dtype=np.float64))

if _NUMBA:
    @_njit(parallel=True, cache=False, fastmath=False)
    def _jit_perturb(dcr_arr, dci_arr, W, H, max_iter,
                     ref_zr, ref_zi, ref_len):
        """Perturbation delta loop with glitch flagging.

        δZ_{n+1} = 2·Z_ref[n]·δZ_n  +  δZ_n²  +  δC

        Glitch detection (Pauldelbrot criterion):
          |Z_ref[n]| < 1e-3  →  reference is near zero, perturbation
          approximation breaks down. Flag pixel as -1.0 (sentinel).

        Glitched pixels are collected by the Python caller and re-rendered
        with a fresh reference orbit chosen from within the glitched region.
        This is the standard multi-reference approach used by Kalles Fraktaler.

        fastmath=False: every float64 bit counts here.
        """
        out  = np.zeros((H, W), dtype=np.float64)
        log2 = _math.log(2.0)
        GLITCH_SENTINEL = -1.0
        for row in _prange(H):
            dci = dci_arr[row]
            for col in range(W):
                dcr = dcr_arr[col]
                dr = 0.0;  di = 0.0
                glitched = False
                n = 0
                while n < ref_len - 1:
                    Zr = ref_zr[n];  Zi = ref_zi[n]
                    # Glitch: reference orbit too close to zero
                    # perturbation formula 2·Z_ref·δZ loses all info
                    ref_mag2 = Zr*Zr + Zi*Zi
                    if ref_mag2 < 1e-3:
                        glitched = True
                        break
                    # δZ_{n+1} = 2·Z_ref·δZ + δZ² + δC
                    new_dr = 2.0*(Zr*dr - Zi*di) + (dr*dr - di*di) + dcr
                    new_di = 2.0*(Zr*di + Zi*dr) + (2.0*dr*di)     + dci
                    dr = new_dr;  di = new_di
                    # Escape check: full Z = Z_ref[n+1] + δZ
                    Fr = ref_zr[n+1] + dr
                    Fi = ref_zi[n+1] + di
                    mag2 = Fr*Fr + Fi*Fi
                    if mag2 > 4.0:
                        log_zn = _math.log(max(mag2, 1e-300)) / 2.0
                        nu     = _math.log(max(log_zn / log2, 1e-300)) / log2
                        out[row, col] = max(0.0, n + 1.0 - nu)
                        break
                    # Secondary glitch: δZ has grown larger than Z_ref
                    # approximation is no longer valid
                    delt_mag2 = dr*dr + di*di
                    if delt_mag2 > ref_mag2:
                        glitched = True
                        break
                    n += 1
                if glitched:
                    out[row, col] = GLITCH_SENTINEL
        return out


def compute_perturb_mandelbrot(xmin, xmax, ymin, ymax, W, H, max_iter, zoom):
    """Perturbation render for extreme zoom Mandelbrot.

    Pixel size is computed directly as a single small mpmath number:
        pixel_size = 3.5 / (zoom * W)
    Never derived from (xmax - xmin) — that subtraction destroys precision
    when xmin and xmax are large floats differing only in low-order bits.

    δC for pixel (col, row):
        dcr = (col - W/2) * pixel_size
        dci = (row - H/2) * pixel_size_h
    Pure integer offsets from center, scaled by one tiny number. No
    absolute coordinates anywhere except to locate the center point.
    """
    if not _MPMATH:
        return _jit_mandelbrot(xmin, xmax, ymin, ymax, W, H, max_iter)

    # Write to a log file so it's visible regardless of terminal state
    import sys, pathlib
    _log = pathlib.Path.home() / "fractal_perturb.log"
    with open(_log, "a") as _f:
        _f.write(f"[PERTURB] zoom={zoom:.3e}  prec={_mp.mp.prec}b  numba={_NUMBA}\n")

    # Set precision floor at 200 bits (~60 decimal digits).
    # Scale up with zoom: each decade needs ~3.32 extra bits.
    prec = max(200, int(_math.log10(max(zoom, 1.0)) * 4) + 200)
    _mp.mp.prec = prec

    # Center — only absolute coord use. float64 repr gives ~16 sig digits,
    # enough to locate the center; pixel offsets handle the rest.
    cx_mp = (_mp.mpf(repr(xmin)) + _mp.mpf(repr(xmax))) / 2
    cy_mp = (_mp.mpf(repr(ymin)) + _mp.mpf(repr(ymax))) / 2

    # Pixel size as a single tiny mpmath number — never subtract large floats
    pixel_w = _mp.mpf('3.5') / (_mp.mpf(repr(zoom)) * W)
    pixel_h = _mp.mpf('3.5') / (_mp.mpf(repr(zoom)) * H)

    # Reference orbit — computed ONCE at full mpmath precision before pixel loop
    ref_zr, ref_zi = _reference_orbit(cx_mp, cy_mp, max_iter)
    ref_len = len(ref_zr)

    # Search for a long-lived reference point.
    # A good reference must survive close to max_iter iterations.
    # Strategy: spiral outward from center across multiple scales.
    if ref_len < max_iter * 3 // 4:
        best_len = ref_len
        best_ref = (cx_mp, cy_mp, ref_zr, ref_zi)
        # Search at multiple scales: 1/8, 1/4, 1/2 of frame
        for _scale in [0.125, 0.25, 0.5, 1.0]:
            sw = pixel_w * W * _scale
            sh = pixel_h * H * _scale
            offsets = []
            steps = 5
            for _i in range(-steps, steps+1):
                for _j in range(-steps, steps+1):
                    if _i == 0 and _j == 0:
                        continue
                    offsets.append((sw * _i / steps, sh * _j / steps))
            for dr_off, di_off in offsets:
                cand_zr, cand_zi = _reference_orbit(
                    cx_mp + dr_off, cy_mp + di_off, max_iter)
                if len(cand_zr) > best_len:
                    best_len = len(cand_zr)
                    best_ref = (cx_mp+dr_off, cy_mp+di_off, cand_zr, cand_zi)
            if best_len >= max_iter * 3 // 4:
                break  # good enough — stop searching
        cx_mp, cy_mp, ref_zr, ref_zi = best_ref
        ref_len = len(ref_zr)
    _log = _pathlib.Path.home() / "fractal_perturb.log"
    with open(_log, "a") as _f:
        _f.write(f"  [REF] final ref_len={ref_len} max_iter={max_iter}\n")

    # δC arrays: integer pixel offsets from center × pixel_size
    # (col - W/2) is a small integer — no float precision lost here
    half_W = W / 2.0;  half_H = H / 2.0
    dcr_arr = np.array(
        [float((_mp.mpf(c) - half_W) * pixel_w) for c in range(W)],
        dtype=np.float64)
    dci_arr = np.array(
        [float((_mp.mpf(r) - half_H) * pixel_h) for r in range(H)],
        dtype=np.float64)

    if _NUMBA:
        try:
            out = _jit_perturb(dcr_arr, dci_arr, W, H, max_iter,
                               ref_zr, ref_zi, ref_len)
        except Exception as _jit_err:
            _log = _pathlib.Path.home() / "fractal_perturb.log"
            with open(_log, "a") as _f:
                _f.write(f"  [JIT-ERROR] {type(_jit_err).__name__}: {_jit_err}\n")
            import traceback
            with open(_log, "a") as _f:
                _f.write(traceback.format_exc() + "\n")
            raise

        # Diagnostic: log sentinel count immediately after first pass
        _n_sentinels = int(np.sum(out == -1.0))
        _n_zeros = int(np.sum(out == 0.0))
        _n_pos = int(np.sum(out > 0.0))
        _log = _pathlib.Path.home() / "fractal_perturb.log"
        with open(_log, "a") as _f:
            _f.write(f"  [POST-JIT] sentinels={_n_sentinels} zeros={_n_zeros} "
                     f"escaped={_n_pos} total={W*H} ref_len={ref_len}\n")
        # Safety: if 100% glitched, the perturbation approximation has
        # completely broken down. Fall back to float64 JIT directly —
        # it will be imprecise but won't be black.
        if _n_sentinels == W * H:
            _log = _pathlib.Path.home() / "fractal_perturb.log"
            with open(_log, "a") as _f:
                _f.write(f"  [FALLBACK] 100% glitch — using float64 JIT\n")
            out = _jit_mandelbrot(xmin, xmax, ymin, ymax, W, H, max_iter)

        # ── Multi-reference pass: fix glitched pixels ─────────────────────
        MAX_GLITCH_PASSES = 8
        for _pass in range(MAX_GLITCH_PASSES):
            glitch_mask = (out == -1.0)
            n_glitch = int(np.sum(glitch_mask))
            if n_glitch == 0:
                break
            _log = _pathlib.Path.home() / "fractal_perturb.log"
            with open(_log, "a") as _f:
                _f.write(f"  [GLITCH pass {_pass+1}] {n_glitch}/{W*H} pixels\n")

            # If > 80% of pixels are glitched, the primary reference is bad.
            # Search a grid of candidates across the frame for the longest orbit.
            if n_glitch > W * H * 0.8:
                best_len = 0
                best_cx, best_cy = cx_mp, cy_mp
                for _sr in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    for _sc in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        _tcx = cx_mp + (_mp.mpf(_sc - 0.5) * W) * pixel_w
                        _tcy = cy_mp + (_mp.mpf(_sr - 0.5) * H) * pixel_h
                        _tzr, _tzi = _reference_orbit(_tcx, _tcy, max_iter)
                        if len(_tzr) > best_len:
                            best_len = len(_tzr)
                            best_cx, best_cy = _tcx, _tcy
                            best_ref_zr, best_ref_zi = _tzr, _tzi
                _log = _pathlib.Path.home() / "fractal_perturb.log"
                with open(_log, "a") as _f:
                    _f.write(f"  [GRID-SEARCH] best_ref_len={best_len}\n")
                if best_len >= 4:
                    gc_f = float((best_cx - cx_mp) / pixel_w + W/2)
                    gr_f = float((best_cy - cy_mp) / pixel_h + H/2)
                    gc, gr = int(round(gc_f)), int(round(gr_f))
                    new_cx, new_cy = best_cx, best_cy
                    new_ref_zr, new_ref_zi = best_ref_zr, best_ref_zi
                    new_ref_len = best_len
                else:
                    out[out == -1.0] = 0.0
                    break
            else:
                # Pick a glitched pixel near the center of the glitch cluster
                rows_g, cols_g = np.where(glitch_mask)
                mid = len(rows_g) // 2
                gr, gc = int(rows_g[mid]), int(cols_g[mid])
                new_cx = cx_mp + (_mp.mpf(gc) - W/2) * pixel_w
                new_cy = cy_mp + (_mp.mpf(gr) - H/2) * pixel_h
                new_ref_zr, new_ref_zi = _reference_orbit(new_cx, new_cy, max_iter)
                new_ref_len = len(new_ref_zr)
                if new_ref_len < 4:
                    out[out == -1.0] = 0.0
                    break
            # δC for each glitched pixel relative to NEW reference
            # δC relative to new reference = (pixel_pos - new_ref_pos)
            # = integer offset from the glitch pixel × pixel_size
            # This is exact — no large float subtraction
            new_dcr = np.array(
                [float((_mp.mpf(c) - gc) * pixel_w) for c in range(W)],
                dtype=np.float64)
            new_dci = np.array(
                [float((_mp.mpf(r) - gr) * pixel_h) for r in range(H)],
                dtype=np.float64)
            out2 = _jit_perturb(new_dcr, new_dci, W, H, max_iter,
                                new_ref_zr, new_ref_zi, new_ref_len)
            # Only copy non-glitch results back
            fixed_mask = glitch_mask & (out2 != -1.0)
            out[fixed_mask] = out2[fixed_mask]
            # Remaining -1s that out2 also glitched: leave as 0 (inside set)
            still_glitch = glitch_mask & (out2 == -1.0)
            out[still_glitch] = 0.0

        # Any residual -1 sentinels → treat as interior (iteration count 0)
        out[out == -1.0] = 0.0
        return out

    # numpy fallback — same logic, no JIT
    out  = np.zeros((H, W), dtype=np.float64)
    import math;  log2 = math.log(2.0)
    for row in range(H):
        dci = dci_arr[row]
        for col in range(W):
            dcr = dcr_arr[col]
            dr = di = 0.0
            for n in range(ref_len - 1):
                Zr = ref_zr[n];  Zi = ref_zi[n]
                new_dr = 2*(Zr*dr - Zi*di) + (dr*dr - di*di) + dcr
                new_di = 2*(Zr*di + Zi*dr) + (2*dr*di)        + dci
                dr = new_dr;  di = new_di
                # Z_ref + δZ only at escape check
                Fr = ref_zr[n+1] + dr;  Fi = ref_zi[n+1] + di
                mag2 = Fr*Fr + Fi*Fi
                if mag2 > 4.0:
                    log_zn = math.log(max(mag2, 1e-300)) / 2.0
                    nu     = math.log(max(log_zn / log2, 1e-300)) / log2
                    out[row, col] = max(0.0, n + 1.0 - nu)
                    break
    return out

if _MLX:
    def _mlx_fractal(xmin, xmax, ymin, ymax, W, H, max_iter, mode, jcr=0.0, jci=0.0):
        """Run fractal on M4 GPU via MLX (float32 only — Metal limitation)."""
        re = _mx.array(np.linspace(xmin, xmax, W, dtype=np.float32))
        im = _mx.array(np.linspace(ymin, ymax, H, dtype=np.float32))
        CR = _mx.broadcast_to(re[None, :], (H, W))
        CI = _mx.broadcast_to(im[:, None], (H, W))

        if mode == "julia":
            ZR = CR; ZI = CI
            CR_c = _mx.full((H, W), float(jcr), dtype=_mx.float32)
            CI_c = _mx.full((H, W), float(jci), dtype=_mx.float32)
        else:
            ZR = _mx.zeros((H, W), dtype=_mx.float32)
            ZI = _mx.zeros((H, W), dtype=_mx.float32)
            CR_c = CR
            CI_c = CI

        count   = _mx.zeros((H, W), dtype=_mx.float32)
        escaped = _mx.zeros((H, W), dtype=_mx.bool_)
        log2    = float(_math.log(2.0))

        for i in range(max_iter):
            ZR2 = ZR * ZR
            ZI2 = ZI * ZI
            mag2 = ZR2 + ZI2
            new_esc = (~escaped) & (mag2 > 4.0)
            safe_mag2 = _mx.where(new_esc, _mx.maximum(mag2, 1e-10), _mx.ones_like(mag2))
            log_zn = _mx.log(safe_mag2) / 2.0
            nu = _mx.log(_mx.maximum(log_zn / log2, 1e-10)) / log2
            smooth_i = _mx.maximum(_mx.zeros_like(count), (i + 1.0) - nu)
            count = _mx.where(new_esc, smooth_i, count)
            escaped = escaped | new_esc
            still = ~escaped
            if mode == "burning_ship":
                new_ZI = 2.0 * _mx.abs(ZR) * _mx.abs(ZI) + CI_c
                new_ZR = ZR2 - ZI2 + CR_c
            elif mode == "tricorn":
                new_ZI = -2.0 * ZR * ZI + CI_c
                new_ZR = ZR2 - ZI2 + CR_c
            else:
                new_ZI = 2.0 * ZR * ZI + CI_c
                new_ZR = ZR2 - ZI2 + CR_c
            ZR = _mx.where(still, new_ZR, ZR)
            ZI = _mx.where(still, new_ZI, ZI)
            if i % 32 == 31:
                _mx.eval(escaped)
                if escaped.all():
                    break

        _mx.eval(count)
        return np.array(count, dtype=np.float64)

    # Zoom threshold: above this float32 loses precision, fall back to numba/numpy
    _MLX_ZOOM_LIMIT = 800.0

    def compute_escape(mode, C, max_iter, smooth=True, xmin=0,xmax=0,ymin=0,ymax=0,W=0,H=0,zoom=0):
        if W and H:
            # Use caller-supplied zoom (from state accumulator) when available
            # — avoids catastrophic cancellation from xmax-xmin at extreme depth
            if not zoom:
                _span = abs(xmax - xmin)
                zoom = 3.5 / _span if _span > 1e-300 else 1e13
            if zoom < _MLX_ZOOM_LIMIT:
                return _mlx_fractal(xmin,xmax,ymin,ymax,W,H,max_iter,mode)
            elif zoom >= _PERTURB_ZOOM_LIMIT and mode == "mandelbrot":
                return compute_perturb_mandelbrot(xmin,xmax,ymin,ymax,W,H,max_iter,zoom)
            elif _NUMBA:
                if mode=="mandelbrot":    return _jit_mandelbrot(xmin,xmax,ymin,ymax,W,H,max_iter)
                elif mode=="burning_ship":return _jit_burning_ship(xmin,xmax,ymin,ymax,W,H,max_iter)
                elif mode=="tricorn":     return _jit_tricorn(xmin,xmax,ymin,ymax,W,H,max_iter)
        return _numpy_escape(mode, C, max_iter)

    def compute_julia(C, jc, max_iter, smooth=True, xmin=0,xmax=0,ymin=0,ymax=0,W=0,H=0):
        if W and H:
            zoom = 3.5 / max(abs(xmax - xmin), 1e-300)
            if zoom < _MLX_ZOOM_LIMIT:
                return _mlx_fractal(xmin,xmax,ymin,ymax,W,H,max_iter,"julia",jc.real,jc.imag)
            # Deep zoom: numba float64 handles arbitrary depth
            if _NUMBA:
                return _jit_julia(xmin,xmax,ymin,ymax,W,H,max_iter,jc.real,jc.imag)
            # numpy fallback with local C grid (caller may pass None)
            x=np.linspace(xmin,xmax,W); y=np.linspace(ymin,ymax,H)
            C=x[np.newaxis,:]+1j*y[:,np.newaxis]
        if C is None:
            x=np.linspace(xmin,xmax,max(W,2)); y=np.linspace(ymin,ymax,max(H,2))
            C=x[np.newaxis,:]+1j*y[:,np.newaxis]
        return _numpy_julia(C, jc, max_iter)

# ── numba CPU fallback (JIT fns already defined above) ───────────────────────
elif _NUMBA:
    def compute_escape(mode,C,max_iter,smooth=True,xmin=0,xmax=0,ymin=0,ymax=0,W=0,H=0,zoom=0):
        if W and H:
            if not zoom:
                zoom = 3.5 / max(abs(xmax - xmin), 1e-300)
            if zoom >= _PERTURB_ZOOM_LIMIT and mode == "mandelbrot":
                return compute_perturb_mandelbrot(xmin,xmax,ymin,ymax,W,H,max_iter,zoom)
            if mode=="mandelbrot":    return _jit_mandelbrot(xmin,xmax,ymin,ymax,W,H,max_iter)
            elif mode=="burning_ship":return _jit_burning_ship(xmin,xmax,ymin,ymax,W,H,max_iter)
            elif mode=="tricorn":     return _jit_tricorn(xmin,xmax,ymin,ymax,W,H,max_iter)
        return _numpy_escape(mode,C,max_iter)

    def compute_julia(C,jc,max_iter,smooth=True,xmin=0,xmax=0,ymin=0,ymax=0,W=0,H=0):
        if W and H: return _jit_julia(xmin,xmax,ymin,ymax,W,H,max_iter,jc.real,jc.imag)
        return _numpy_julia(C,jc,max_iter)

# ── numpy pure fallback ───────────────────────────────────────────────────────
else:
    def compute_escape(mode,C,max_iter,smooth=True,**kw):
        return _numpy_escape(mode,C,max_iter)
    def compute_julia(C,jc,max_iter,smooth=True,**kw):
        return _numpy_julia(C,jc,max_iter)

def _numpy_escape(mode,C,max_iter):
    Z=np.zeros_like(C,dtype=np.complex128)
    count=np.zeros(C.shape,dtype=np.float64); mask=np.ones(C.shape,dtype=bool)
    for i in range(max_iter):
        if not np.any(mask): break
        if mode=="mandelbrot": Z[mask]=Z[mask]**2+C[mask]
        elif mode=="burning_ship":
            Zr=np.abs(Z[mask].real)+1j*np.abs(Z[mask].imag); Z[mask]=Zr**2+C[mask]
        elif mode=="tricorn": Z[mask]=np.conj(Z[mask])**2+C[mask]
        mag2=Z.real**2+Z.imag**2; esc=mask&(mag2>4.0)
        if np.any(esc):
            log_zn=np.log(np.maximum(mag2[esc],1e-10))/2
            nu=np.log(np.maximum(log_zn/np.log(2),1e-10))/np.log(2)
            count[esc]=np.maximum(0,i+1-nu)
        mask[esc]=False
    return count

def _numpy_julia(C,jc,max_iter):
    Z=C.copy().astype(np.complex128)
    count=np.zeros(C.shape,dtype=np.float64); mask=np.ones(C.shape,dtype=bool)
    for i in range(max_iter):
        if not np.any(mask): break
        Z[mask]=Z[mask]**2+jc; mag2=Z.real**2+Z.imag**2; esc=mask&(mag2>4.0)
        if np.any(esc):
            log_zn=np.log(np.maximum(mag2[esc],1e-10))/2
            nu=np.log(np.maximum(log_zn/np.log(2),1e-10))/np.log(2)
            count[esc]=np.maximum(0,i+1-nu)
        mask[esc]=False
    return count

def dynamic_tui_iters(max_iter, zoom):
    import math
    # Floor: minimum iters needed for meaningful TUI preview at this depth
    if zoom >= 1e18:   floor = 3000
    elif zoom >= 1e15: floor = 2000
    elif zoom >= 1e12: floor = 1500
    elif zoom >= 1e9:  floor = 1000
    elif zoom >= 1e6:  floor = 600
    elif zoom >= 1e4:  floor = 500
    elif zoom >= 1e3:  floor = 400
    elif zoom >= 100:  floor = 256
    elif zoom >= 10:   floor = 128
    else:              floor = 64
    # Always honour the user's max_iter if they've set it higher than the floor
    return max(max_iter, floor)

def count_to_rgb(count, max_iter, palette):
    interior = count == 0
    exterior = ~interior

    # Histogram equalization for deep zoom: redistributes colors so the full
    # palette is used regardless of how tightly clustered iteration counts are.
    t = np.zeros(count.shape, dtype=np.float64)
    if exterior.any():
        ext_vals = count[exterior]
        # Build cumulative histogram over exterior pixels only
        min_v, max_v = ext_vals.min(), ext_vals.max()
        if max_v > min_v:
            # 2048-bin histogram for smooth gradients
            hist, edges = np.histogram(ext_vals, bins=2048,
                                       range=(min_v, max_v))
            cdf = np.cumsum(hist).astype(np.float64)
            cdf /= cdf[-1]  # normalize to [0, 1]
            # Map each exterior pixel's count to its CDF value
            bin_idx = np.clip(
                ((ext_vals - min_v) / (max_v - min_v) * 2047).astype(np.int32),
                0, 2047)
            t[exterior] = cdf[bin_idx]
        else:
            t[exterior] = 0.5  # all same count — flat field

    rgb = apply_palette(palette, t)
    rgb[interior] = [0, 0, 0]
    return rgb

# ── Block renderer ────────────────────────────────────────────────────────────
def rgb_to_blocks(rgb):
    H,W,_=rgb.shape
    if H%2==1:
        rgb=np.vstack([rgb,np.zeros((1,W,3),dtype=np.uint8)]); H+=1
    lines=[]
    for row in range(0,H,2):
        top=rgb[row]; bot=rgb[row+1]; parts=[]
        for col in range(W):
            tr,tg,tb=top[col]; br,bg_,bb=bot[col]
            parts.append(
                f"\x1b[38;2;{tr};{tg};{tb}m"
                f"\x1b[48;2;{br};{bg_};{bb}m▀\x1b[0m"
            )
        lines.append("".join(parts))
    return lines

def draw_orbit_blocks(lines, pts, esc, state, cols, rows):
    """Overlay orbit dots on block-render lines (mutates list in place)."""
    if not pts: return
    xmin,xmax=state["xmin"],state["xmax"]
    ymin,ymax=state["ymin"],state["ymax"]
    color = "\x1b[38;2;255;80;80m" if esc else "\x1b[38;2;80;255;80m"
    H_chars = len(lines)
    for zr,zi in pts:
        col = int((zr-xmin)/(xmax-xmin)*cols)
        # each line = 2 pixel rows
        row = int((zi-ymin)/(ymax-ymin)*H_chars)
        if 0<=row<H_chars and 0<=col<cols:
            line = lines[row]
            # insert colored dot at col position (approximate — ANSI aware replace is complex)
            # We rebuild just that character
            lines[row] = line  # leave as-is; dots will show on blast

# ── iTerm2 blast ──────────────────────────────────────────────────────────────
def get_terminal_pixel_size():
    """Query iTerm2 for actual pixel dimensions via XTWINOPS."""
    import struct, fcntl, termios
    try:
        buf = b'\x00'*8
        result = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, buf)
        rows, cols, xpix, ypix = struct.unpack('HHHH', result)
        if xpix > 0 and ypix > 0:
            return xpix, ypix
    except Exception:
        pass
    # fallback: estimate from cell count
    t = blessed.Terminal()
    return t.width * 8, t.height * 16

def iterm2_blast(img):
    buf=io.BytesIO(); img.save(buf,format="PNG")
    data=base64.b64encode(buf.getvalue()).decode(); size=len(buf.getvalue())
    sys.stdout.write(
        f"\x1b]1337;File=inline=1;size={size};"
        f"width={img.width}px;height={img.height}px;"
        f"preserveAspectRatio=1:{data}\a\n"
    )
    sys.stdout.flush()

def build_hires(state, orbit=None):
    W,H = get_terminal_pixel_size()
    # leave a few px for status lines
    H = max(400, H - 60)
    W = max(600, W)

    xmin,xmax,ymin,ymax=state["xmin"],state["xmax"],state["ymin"],state["ymax"]
    if _MLX or _NUMBA:
        if state["julia_mode"]:
            count = compute_julia(None, state["julia_c"], state["max_iter"],
                                  xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,W=W,H=H)
        else:
            count = compute_escape(state["mode"], None, state["max_iter"],
                                   xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,W=W,H=H,
                                   zoom=state["zoom"])
    else:
        x=np.linspace(xmin,xmax,W); y=np.linspace(ymin,ymax,H)
        C=x[np.newaxis,:]+1j*y[:,np.newaxis]
        count=(compute_julia(C,state["julia_c"],state["max_iter"])
               if state["julia_mode"]
               else compute_escape(state["mode"],C,state["max_iter"]))
    rgb=count_to_rgb(count,state["max_iter"],state["palette"])
    rgb[::2]=(rgb[::2]*0.82).astype(np.uint8)  # scanlines
    img=Image.fromarray(rgb)

    # Orbit overlay — clip to image bounds for speed, skip runaway escaped pts
    if orbit:
        pts,esc=orbit; draw=ImageDraw.Draw(img)
        color="#ff5050" if esc else "#50ff50"
        def to_px(zr,zi):
            return int((zr-xmin)/(xmax-xmin)*W),int((zi-ymin)/(ymax-ymin)*H)
        # Only draw pts that are within or near the image bounds (clip runaway orbit)
        margin = max(W, H) * 2
        scr=[to_px(zr,zi) for zr,zi in pts]
        visible = [(px,py) for (px,py) in scr if -margin<=px<=W+margin and -margin<=py<=H+margin]
        for i in range(len(visible)-1):
            draw.line([visible[i],visible[i+1]],fill=color,width=1)
        for i,(px,py) in enumerate(visible[:min(len(visible),500)]):
            if 0<=px<W and 0<=py<H:
                r=5 if i==0 else 2
                draw.ellipse([(px-r,py-r),(px+r,py+r)],fill=color)

    # Julia preview inset
    if state["julia_preview"] and not state["julia_mode"]:
        jS=min(220, W//5)
        jx=np.linspace(-1.5,1.5,jS); jy=np.linspace(-1.5,1.5,jS)
        jC=jx[np.newaxis,:]+1j*jy[:,np.newaxis]
        jrgb=count_to_rgb(compute_julia(jC,state["julia_c"],128),128,state["palette"])
        jimg=Image.fromarray(jrgb); draw=ImageDraw.Draw(img)
        pad=12; x0,y0=W-jS-pad, H-jS-pad-24
        draw.rectangle([(x0-2,y0-18),(x0+jS+2,y0+jS+2)],outline=(255,191,0),width=1)
        try: font=ImageFont.truetype("/System/Library/Fonts/Menlo.ttc",11)
        except: font=ImageFont.load_default()
        draw.text((x0,y0-16),"JULIA PREVIEW",font=font,fill=(255,191,0))
        jc=state["julia_c"]
        draw.text((x0,y0+jS+4),f"c={jc.real:.4f}+{jc.imag:.4f}i",font=font,fill=(120,120,40))
        img.paste(jimg,(x0,y0))

    return img

# ── State helpers ─────────────────────────────────────────────────────────────
def make_state(mode="mandelbrot"):
    xmin,xmax,ymin,ymax=DEFAULT_COORDS[mode]
    return dict(mode=mode,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,zoom=1.0,
                max_iter=512,palette="hybrid",julia_mode=False,
                julia_preview=True,julia_c=complex(-0.7269,0.1889))

def zoom_at(state, cx_complex, cy_complex, factor):
    """Zoom in/out centered on a specific complex point.
    Zoom is tracked as a separate accumulator — never derived from coord
    subtraction (xmax-xmin collapses to zero at extreme depth)."""
    hw=(state["xmax"]-state["xmin"])/2/factor
    hh=(state["ymax"]-state["ymin"])/2/factor
    state.update(xmin=cx_complex-hw, xmax=cx_complex+hw,
                 ymin=cy_complex-hh, ymax=cy_complex+hh)
    # Accumulate zoom as float — clamp to avoid inf/overflow display
    new_zoom = state["zoom"] * factor
    state["zoom"] = new_zoom if new_zoom < 1e300 else state["zoom"]

def zoom_center(state, factor):
    cx=(state["xmin"]+state["xmax"])/2
    cy=(state["ymin"]+state["ymax"])/2
    zoom_at(state, cx, cy, factor)

def pan_view(state, dx, dy):
    rw=state["xmax"]-state["xmin"]; rh=state["ymax"]-state["ymin"]
    state["xmin"]+=dx*rw; state["xmax"]+=dx*rw
    state["ymin"]+=dy*rh; state["ymax"]+=dy*rh

def reset_view(state):
    xmin,xmax,ymin,ymax=DEFAULT_COORDS[state["mode"]]
    state.update(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,zoom=1.0)

def cell_to_complex(state, col, row, cols, rows):
    """Convert terminal cell (col, row) → complex plane coordinate."""
    # row 0 = header, so subtract 1 for the fractal area
    # each char-row = 2 pixel-rows (▀ block rendering)
    frac_rows = rows - 2   # header + footer
    xmin,xmax=state["xmin"],state["xmax"]
    ymin,ymax=state["ymin"],state["ymax"]
    re = xmin + (col / cols) * (xmax - xmin)
    im = ymin + ((row - 1) / frac_rows) * (ymax - ymin)
    return re, im

def orbit_pts(cr, ci, max_iter, mode, jc=None):
    pts=[]; zr,zi=(cr,ci) if jc else (0.0,0.0)
    c_r,c_i=(jc.real,jc.imag) if jc else (cr,ci)
    pts.append((zr,zi)); esc=False
    for _ in range(max_iter):
        if mode=="mandelbrot":   nzi=2*zr*zi+c_i; nzr=zr*zr-zi*zi+c_r
        elif mode=="burning_ship": nzi=2*abs(zr)*abs(zi)+c_i; nzr=zr*zr-zi*zi+c_r
        else:                    nzi=-2*zr*zi+c_i; nzr=zr*zr-zi*zi+c_r
        zr,zi=nzr,nzi; pts.append((zr,zi))
        if zr*zr+zi*zi>256: esc=True; break
    return pts,esc

# ── TUI ───────────────────────────────────────────────────────────────────────
class FractalTUI:
    def __init__(self):
        self.term   = blessed.Terminal()
        self.state  = make_state()
        self.orbit  = None
        self._dirty = True
        self._status = "o:orbit-center  scroll:zoom  arrows:navigate"

    # ── Block render ──────────────────────────────────────────────────────
    def _block_render(self, cols, rows):
        H_px = max(8, (rows-2)*2)
        W_px = max(8, cols - 34)   # leave 34 cols for sidebar
        xmin,xmax,ymin,ymax = (self.state["xmin"],self.state["xmax"],
                                self.state["ymin"],self.state["ymax"])
        iters = dynamic_tui_iters(self.state["max_iter"], self.state["zoom"])
        if _MLX or _NUMBA:
            if self.state["julia_mode"]:
                count = compute_julia(None, self.state["julia_c"], iters,
                                      xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,W=W_px,H=H_px)
            else:
                count = compute_escape(self.state["mode"], None, iters,
                                       xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,W=W_px,H=H_px,
                                       zoom=self.state["zoom"])
        else:
            x=np.linspace(xmin,xmax,W_px); y=np.linspace(ymin,ymax,H_px)
            C=x[np.newaxis,:]+1j*y[:,np.newaxis]
            count=(compute_julia(C,self.state["julia_c"],iters)
                   if self.state["julia_mode"]
                   else compute_escape(self.state["mode"],C,iters))
        return rgb_to_blocks(count_to_rgb(count,iters,self.state["palette"]))

    # ── CRT color helpers ─────────────────────────────────────────────────
    def _crt(self, text, fg="bright_red", bg=None):
        """Apply RTA-matching CRT phosphor styling via ANSI codes."""
        t = self.term
        CRT_FG   = "\x1b[38;2;255;59;59m"    # #ff3b3b  red phosphor
        CRT_BG   = "\x1b[48;2;18;0;0m"       # #120000  near-black header
        CRT_DIM  = "\x1b[38;2;122;10;10m"    # #7a0a0a  dim border red
        CRT_SEP  = "\x1b[38;2;60;0;0m"       # separator
        BOLD     = "\x1b[1m"
        RESET    = "\x1b[0m"
        if bg == "dark":
            return f"{BOLD}{CRT_FG}{CRT_BG}{text}{RESET}"
        elif bg == "dim":
            return f"{CRT_DIM}{CRT_BG}{text}{RESET}"
        return f"{BOLD}{CRT_FG}{text}{RESET}"

    # ── Header ────────────────────────────────────────────────────────────
    def _header(self, cols):
        t=self.term; s=self.state
        # Use accumulated zoom value — coord subtraction loses precision at depth
        zoom = s["zoom"]
        # Guard against inf/nan from float overflow
        if not (1.0 <= zoom < 1e300):
            zoom = 3.5 / max(abs(s["xmax"] - s["xmin"]), 1e-300)
            if not (1.0 <= zoom < 1e300):
                zoom = 1.0
            s["zoom"] = zoom
        zstr=(f"{zoom/1e15:.3f}Q×" if zoom>=1e15
              else f"{zoom/1e12:.3f}T×" if zoom>=1e12
              else f"{zoom/1e9:.2f}B×" if zoom>=1e9
              else f"{zoom/1e6:.2f}M×" if zoom>=1e6
              else f"{zoom/1000:.1f}K×" if zoom>=1000
              else f"{zoom:.1f}×")
        tags=""
        if s["julia_mode"]: tags+=" \x1b[38;2;255;191;0m[JULIA]\x1b[0m\x1b[1m\x1b[38;2;255;59;59m\x1b[48;2;18;0;0m"
        if self.orbit:      tags+=" \x1b[38;2;55;255;127m[ORBIT]\x1b[0m\x1b[1m\x1b[38;2;255;59;59m\x1b[48;2;18;0;0m"

        sep   = " \x1b[38;2;90;10;10m│\x1b[38;2;255;59;59m "
        left  = (f" \x1b[1m\x1b[38;2;255;59;59m\x1b[48;2;18;0;0m"
                 f"{FRACTAL_EMOJI[s['mode']]} {FRACTAL_LABELS[s['mode']]}{tags}"
                 f"{sep}{zstr}{sep}{s['max_iter']}it{sep}{PALETTE_LABELS[s['palette']]}")
        right = f"[{s['xmin']:.4f}, {s['xmax']:.4f}] \x1b[38;2;90;10;10m×\x1b[38;2;255;59;59m [{s['ymin']:.4f}, {s['ymax']:.4f}] \x1b[0m"

        # Build plain version for length calculation
        plain_left  = f" {FRACTAL_EMOJI[s['mode']]} {FRACTAL_LABELS[s['mode']]}  {zstr}  {s['max_iter']}it  {PALETTE_LABELS[s['palette']]}"
        plain_right = f"[{s['xmin']:.4f}, {s['xmax']:.4f}] × [{s['ymin']:.4f}, {s['ymax']:.4f}] "
        pad = max(0, cols - len(plain_left) - len(plain_right) - 2)
        return left + " " * pad + right

    # ── Footer ────────────────────────────────────────────────────────────
    def _footer(self, cols):
        # RTA-style: dim red bg, bright red text, key highlights in amber
        BG    = "\x1b[48;2;10;0;0m"          # #0a0000 footer bg
        FG    = "\x1b[38;2;122;10;10m"       # #7a0a0a dim red text
        KEY   = "\x1b[1m\x1b[38;2;255;59;59m"  # #ff3b3b bright key names
        AMB   = "\x1b[38;2;255;191;0m"       # #ffbf00 amber for BLAST
        RESET = "\x1b[0m"

        def k(key, label):
            return f"{KEY}{key}{FG}:{label}"

        sep = f"  {FG}·{FG}  "

        status_part = f"{KEY}{self._status}{FG}" if self._status else ""

        keys = (f"{k('wasd','pan')}{sep}{k('z/x','zoom')}{sep}"
                f"{k('m','mode')}{sep}{k('p','pal')}{sep}"
                f"{k('j','julia')}{sep}{k('o','orbit')}{sep}"
                f"{k('[/]','iters')}{sep}{k('1-9','preset')}{sep}"
                f"{k('0','reset')}{sep}"
                f"{AMB}\x1b[1mr{FG}:{AMB}BLAST{RESET}{BG}{FG}{sep}"
                f"{k('S','save')}{sep}{k('q','quit')}")

        if status_part:
            bar = f" {status_part}  {FG}│  {keys} "
        else:
            bar = f" {keys} "

        # Plain version for padding
        plain = f" {self._status}  |  wasd:pan  z/x:zoom  m:mode  p:pal  j:julia  o:orbit  [/]:iters  1-9:preset  0:reset  r:BLAST  S:save  q:quit "
        pad = max(0, cols - len(plain))
        return f"{BG}{FG}{bar}{' ' * pad}{RESET}"

    # ── Sidebar ───────────────────────────────────────────────────────────
    def _sidebar(self, rows):
        """Return a list of `rows` ANSI-styled strings, each exactly 32 chars wide."""
        s = self.state
        W = 32

        # ── Colors ────────────────────────────────────────────────────────
        BG   = "\x1b[48;2;25;0;0m"          # #190000
        BDR  = "\x1b[38;2;122;10;10m"       # #7a0a0a heavy border
        DIM  = "\x1b[38;2;160;50;50m"          # readable muted red (was too dark)
        CRT  = "\x1b[1m\x1b[38;2;255;59;59m"  # #ff3b3b bright phosphor
        AMB  = "\x1b[38;2;255;191;0m"       # #ffbf00 amber
        GRN  = "\x1b[38;2;55;255;127m"      # orbit trapped
        RED2 = "\x1b[38;2;255;80;80m"       # orbit escaped
        RST  = "\x1b[0m"

        def line(content_plain, content_ansi):
            """Pad content_plain to W-4 chars (inside borders), wrap with border."""
            inner = W - 4   # 2 border + 1 space each side
            pad = max(0, inner - len(content_plain))
            return f"{BG}{BDR}║{RST}{BG} {content_ansi}{' '*pad} {BDR}║{RST}"

        def sep(char="─"):
            return f"{BG}{BDR}╠{'═'*(W-2)}╣{RST}"

        def blank():
            return f"{BG}{BDR}║{' '*(W-2)}║{RST}"

        def header():
            txt = "CRT FRACTAL"
            pad = (W-2 - len(txt)) // 2
            return f"{BG}{BDR}╔{'═'*(W-2)}╗{RST}"

        def top():
            return f"{BG}{BDR}╔{'═'*(W-2)}╗{RST}"

        def bot():
            return f"{BG}{BDR}╚{'═'*(W-2)}╝{RST}"

        def label(txt):
            """Dim label row."""
            inner = W - 4
            pad = max(0, inner - len(txt))
            return f"{BG}{BDR}║{RST}{BG} {DIM}{txt}{' '*pad} {BDR}║{RST}"

        def val(txt_plain, txt_ansi=None):
            if txt_ansi is None: txt_ansi = f"{CRT}{txt_plain}"
            return line(txt_plain, txt_ansi)

        # ── Zoom string ───────────────────────────────────────────────────
        zoom = 3.5 / max(abs(s["xmax"] - s["xmin"]), 1e-300)
        zstr = (f"{zoom/1e15:.3f}Q×" if zoom >= 1e15
                else f"{zoom/1e12:.3f}T×" if zoom >= 1e12
                else f"{zoom/1e9:.2f}B×" if zoom >= 1e9
                else f"{zoom/1e6:.2f}M×" if zoom >= 1e6
                else f"{zoom/1000:.1f}K×" if zoom >= 1000
                else f"{zoom:.1f}×")

        # ── Engine short label ─────────────────────────────────────────────
        if _MLX and zoom < 800:
            eng_plain = "MLX GPU (float32)"
            eng_ansi  = f"{AMB}MLX GPU {DIM}float32"
        elif _NUMBA:
            eng_plain = "numba JIT (float64)"
            eng_ansi  = f"{CRT}numba {DIM}float64"
        else:
            eng_plain = "numpy (float64)"
            eng_ansi  = f"{DIM}numpy float64"

        # ── Julia/orbit status ─────────────────────────────────────────────
        mode_tag = ""
        if s["julia_mode"]: mode_tag = " [JULIA]"
        mode_plain = FRACTAL_LABELS[s["mode"]] + mode_tag

        jc = s["julia_c"]
        jc_plain = f"{jc.real:+.4f}{jc.imag:+.4f}i"

        # ── Coords ────────────────────────────────────────────────────────
        cx = (s["xmin"]+s["xmax"])/2
        cy = (s["ymin"]+s["ymax"])/2

        # ── Orbit status ──────────────────────────────────────────────────
        if self.orbit:
            pts, esc = self.orbit
            orb_plain = "ESCAPED" if esc else "TRAPPED"
            orb_ansi  = f"{RED2}● ESCAPED" if esc else f"{GRN}● TRAPPED"
            orb_pts   = f"{len(pts)} pts"
        else:
            orb_plain = "none"
            orb_ansi  = f"{DIM}none"
            orb_pts   = ""

        # ── Build rows ────────────────────────────────────────────────────
        sidebar_lines = [
            top(),
            line("CRT FRACTAL EXPLORER", f"{CRT}CRT FRACTAL EXPLORER"),
            sep(),
            blank(),
            label("MODE"),
            val(mode_plain),
            blank(),
            label("FRACTAL"),
            val(FRACTAL_LABELS[s["mode"]]),
            blank(),
            label("ZOOM"),
            val(zstr),
            blank(),
            label("ITERATIONS"),
            val(str(s["max_iter"])),
            blank(),
            label("PALETTE"),
            val(PALETTE_LABELS[s["palette"]]),
            sep(),
            blank(),
            label("CENTER"),
            val(f"re {cx:+.6f}"),
            val(f"im {cy:+.6f}"),
            blank(),
            label("JULIA c"),
            val(jc_plain),
            sep(),
            blank(),
            label("ENGINE"),
            val(eng_plain, eng_ansi),
            blank(),
            label("ORBIT"),
            val(orb_plain, orb_ansi),
        ]
        if orb_pts:
            sidebar_lines.append(val(orb_pts))

        # ── Keys section ──────────────────────────────────────────────────
        KEY = "\x1b[1m\x1b[38;2;255;59;59m"
        def kline(k, desc):
            plain = f"{k:<6}{desc}"
            ansi  = f"{KEY}{k:<6}{RST}{BG}{DIM}{desc}"
            return line(plain, ansi)

        sidebar_lines += [
            sep(),
            blank(),
            kline("↑↓←→", "pan"),
            kline("z/x", "zoom in/out"),
            kline("r", "BLAST render"),
            kline("m", "cycle mode"),
            kline("p", "cycle palette"),
            kline("j", "julia toggle"),
            kline("o", "trace orbit"),
            kline("c", "clear orbit"),
            kline("[/]", "iters ±32"),
            kline("1-9", "presets"),
            kline("0", "reset"),
            kline("S", "save PNG"),
            kline("q", "quit"),
            blank(),
            bot(),
        ]

        # ── Pad or trim to exact row count ────────────────────────────────
        while len(sidebar_lines) < rows:
            sidebar_lines.insert(-1, blank())
        return sidebar_lines[:rows]

    # ── Full redraw ───────────────────────────────────────────────────────
    def redraw(self):
        t = self.term
        cols, rows = t.width, t.height
        fractal_cols = cols - 34

        frac_lines = self._block_render(cols, rows)
        # frac_lines covers rows-2 char rows (header + fractal body, no footer)
        sidebar = self._sidebar(rows - 1)   # -1 for header row

        out = [t.home, self._header(cols), "\n"]
        frac_h = rows - 2   # header + body (no footer now)
        for i in range(frac_h):
            fl = frac_lines[i] if i < len(frac_lines) else ""
            sl = sidebar[i]    if i < len(sidebar)     else ""
            out.append(fl + sl + "\n")

        sys.stdout.write("".join(out))
        sys.stdout.flush()

    # ── Blast ─────────────────────────────────────────────────────────────
    def blast(self, save_path=None):
        t=self.term; s=self.state
        CRT  = "\x1b[1m\x1b[38;2;255;59;59m\x1b[48;2;18;0;0m"
        AMB  = "\x1b[38;2;255;191;0m\x1b[48;2;18;0;0m"
        DIM  = "\x1b[38;2;122;10;10m\x1b[48;2;10;0;0m"
        RST  = "\x1b[0m"
        # Debug: log zoom state at blast time
        import pathlib, math
        _span = abs(s["xmax"] - s["xmin"])
        _derived = 3.5 / _span if _span > 1e-300 else float("inf")
        _log = pathlib.Path.home() / "fractal_perturb.log"
        with open(_log, "a") as _f:
            _f.write(f"[BLAST] state_zoom={s['zoom']:.6e}  derived_zoom={_derived:.6e}  "
                     f"perturb_threshold=1e13  "
                     f"will_perturb={s['zoom'] >= 1e13 and s['mode']=='mandelbrot'}  "
                     f"mpmath={_MPMATH}  numba={_NUMBA}\n")
        sys.stdout.write(t.clear)
        sys.stdout.write(f"\n  {CRT}rendering {FRACTAL_EMOJI[s['mode']]} {FRACTAL_LABELS[s['mode']]}...{RST}\n")
        sys.stdout.flush()
        t0=time.time()
        img=build_hires(s,self.orbit)
        elapsed=time.time()-t0
        iterm2_blast(img)
        if save_path:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(str(save_path))
                save_note = f"  {AMB}✓ saved → {save_path}{RST}\n"
            except Exception as e:
                save_note = f"  {DIM}✗ save failed: {e}{RST}\n"
        else:
            save_note = ""
        W,H=img.size
        derived_zoom = 3.5 / max(abs(s['xmax']-s['xmin']),1e-300)
        sep = f"  {DIM}·{RST}  "
        engine_tag = f"{AMB}MLX{RST}" if _MLX and derived_zoom < 800 else f"{DIM}CPU{RST}"
        zoom_disp = (f"{derived_zoom:.3f}T×" if derived_zoom>=1e12
                     else f"{derived_zoom/1e9:.2f}B×" if derived_zoom>=1e9
                     else f"{derived_zoom/1e6:.2f}M×" if derived_zoom>=1e6
                     else f"{derived_zoom/1000:.1f}K×" if derived_zoom>=1000
                     else f"{derived_zoom:.1f}×")
        sys.stdout.write(
            f"  {CRT}{FRACTAL_EMOJI[s['mode']]}  {W}×{H}{RST}"
            f"{sep}{CRT}zoom {zoom_disp}{RST}"
            f"{sep}{CRT}{s['max_iter']}it{RST}"
            f"{sep}{CRT}{PALETTE_LABELS[s['palette']]}{RST}"
            f"{sep}{engine_tag}"
            f"{sep}{AMB}{elapsed:.2f}s{RST}"
            f"  {DIM}— any key to return{RST}\n"
        )
        if save_note:
            sys.stdout.write(save_note)
        sys.stdout.flush()
        with t.cbreak(): t.inkey()
        sys.stdout.write(t.clear)
        if save_path:
            self._status = f"saved → {save_path.name}"
        self._dirty=True

    # ── Mouse handler (scroll only — clicks owned by iTerm2) ─────────────
    def handle_mouse(self, key):
        t=self.term; cols,rows=t.width,t.height
        try:
            me=key._mode_values
            btn=me.button_value
            if me.released: return
            # Map scroll to zoom centered on viewport center
            cx=(self.state["xmin"]+self.state["xmax"])/2
            cy=(self.state["ymin"]+self.state["ymax"])/2
            if btn==64:   zoom_at(self.state,cx,cy,1.3);  self._status="scroll zoom in";  self._dirty=True
            elif btn==65: zoom_at(self.state,cx,cy,1/1.3);self._status="scroll zoom out"; self._dirty=True
        except Exception:
            pass

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self):
        t=self.term
        engine = _ENGINE
        sys.stdout.write(t.clear)
        sys.stdout.write(f"\n  🦞 CRT FRACTAL EXPLORER  [{engine}]  — first render compiles JIT cache...\n\n")
        sys.stdout.flush()
        time.sleep(0.5)
        try:
            with t.cbreak(), t.hidden_cursor(), t.mouse_enabled(report_drag=False):
                while True:
                    if self._dirty:
                        self.redraw(); self._dirty=False

                    try:
                        key=t.inkey(timeout=0.05)
                    except KeyboardInterrupt:
                        break
                    if not key: continue

                    # Mouse events
                    if key.code==t.KEY_MOUSE:
                        self.handle_mouse(key); continue

                    k=str(key)

                    if k in ('q','Q'): break

                    elif key.code==t.KEY_UP:
                        pan_view(self.state,0,-0.15); self._dirty=True
                    elif key.code==t.KEY_DOWN:
                        pan_view(self.state,0, 0.15); self._dirty=True
                    elif key.code==t.KEY_LEFT:
                        pan_view(self.state,-0.15,0); self._dirty=True
                    elif key.code==t.KEY_RIGHT:
                        pan_view(self.state, 0.15,0); self._dirty=True

                    elif k=='z': zoom_center(self.state,2.0); self._dirty=True
                    elif k=='x': zoom_center(self.state,0.5); self._dirty=True

                    elif k in ('m','M'):
                        idx=FRACTAL_MODES.index(self.state["mode"])
                        self.state["mode"]=FRACTAL_MODES[(idx+1)%len(FRACTAL_MODES)]
                        reset_view(self.state); self.orbit=None; self._dirty=True

                    elif k in ('p','P'):
                        idx=PALETTES.index(self.state["palette"])
                        self.state["palette"]=PALETTES[(idx+1)%len(PALETTES)]
                        self._dirty=True

                    elif k in ('j','J'):
                        self.state["julia_mode"]=not self.state["julia_mode"]
                        if self.state["julia_mode"]:
                            self.state.update(xmin=-1.5,xmax=1.5,ymin=-1.5,ymax=1.5,zoom=1.0)
                            self._status="julia mode — navigate then o to trace orbit"
                        else:
                            reset_view(self.state)
                            self._status="o:orbit-center  scroll:zoom  arrows:navigate"
                        self.orbit=None; self._dirty=True

                    elif k=='c':
                        self.orbit=None
                        self._status="orbit cleared"
                        self._dirty=True

                    elif k=='o':
                        cx=(self.state["xmin"]+self.state["xmax"])/2
                        cy=(self.state["ymin"]+self.state["ymax"])/2
                        jc=self.state["julia_c"] if self.state["julia_mode"] else None
                        pts,esc=orbit_pts(cx,cy,self.state["max_iter"],self.state["mode"],jc)
                        self.orbit=(pts,esc)
                        self._status=(f"orbit @ {cx:.6f}+{cy:.6f}i  "
                                      f"{'ESCAPED 🔴' if esc else 'TRAPPED 🟢'}  "
                                      f"{len(pts)} pts")
                        self._dirty=True

                    elif k=='[':
                        self.state["max_iter"]=max(32,self.state["max_iter"]-1000)
                        self._dirty=True
                    elif k==']':
                        self.state["max_iter"]=min(100000,self.state["max_iter"]+1000)
                        self._dirty=True

                    elif k in ('R',):
                        reset_view(self.state); self.orbit=None; self._dirty=True

                    elif k in PRESETS:
                        name,xmin,xmax,ymin,ymax=PRESETS[k]
                        self.state.update(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
                                          zoom=3.5/(xmax-xmin),julia_mode=False)
                        self.orbit=None
                        self._status=f"preset: {name}"
                        self._dirty=True

                    elif k in ('r','R'): self.blast()

                    elif k=='S':
                        out=pathlib.Path.home()/"Desktop"/f"fractal_{self.state['mode']}_{int(time.time())}.png"
                        self._save_path = out   # blast() will pick this up
                        self.blast(save_path=out)

        finally:
            # Explicitly disable mouse reporting so escape codes don't leak into parent terminal
            sys.stdout.write("\x1b[?1000l")   # disable mouse click reporting
            sys.stdout.write("\x1b[?1002l")   # disable mouse drag reporting
            sys.stdout.write("\x1b[?1003l")   # disable all mouse movement reporting
            sys.stdout.write("\x1b[?1006l")   # disable SGR mouse mode
            sys.stdout.write(t.normal_cursor + t.normal)
            sys.stdout.flush()
            print("\n  👋 bye\n")

if __name__=="__main__":
    FractalTUI().run()
