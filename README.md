# fractal-explorer

A high-performance Mandelbrot explorer that runs entirely in your terminal. Navigate to extreme zoom depths using a four-tier compute engine — MLX GPU → numba JIT → perturbation theory → numpy fallback — with automatic precision switching as you go deeper.

![fractal-explorer hero](screenshots/01_hero_nucleus.png)

*PSYCH palette · deep zoom · MacBook Pro terminal*

---

## What it does

Most fractal renderers break at deep zoom because standard floating point runs out of precision. This one switches engines automatically based on zoom depth:

- **Shallow zoom** — MLX GPU (Apple Silicon unified memory) for maximum throughput
- **Mid zoom** — numba JIT-compiled CPU with parallel pixel loops
- **Deep zoom (>1e13×)** — perturbation theory: compute one reference orbit at arbitrary precision with mpmath, then approximate all surrounding pixels with cheap delta math
- **Fallback** — numpy for environments without MLX or numba

The result: verified renders at **140 trillion× zoom** in under 7 seconds on a MacBook Pro.

---

## Screenshots

![mandala](screenshots/02_mandala.png)
*PSYCH · 17,592,186,044,416× zoom · 79,512 iterations · 11.57s*

---

### Terminal UI

The explorer runs as a full TUI with live navigation, a stats sidebar, and a Julia set preview window. Press `r` at any point to blast a full-resolution render to iTerm2.

![terminal ui clean](screenshots/03_tui_clean.png)
*Live TUI — sidebar shows zoom depth, iteration count, active engine, and Julia parameter in real time*

![terminal ui precision ceiling](screenshots/04_tui_precision_ceiling.png)
*Float64 precision breakdown at 4.4T× zoom — block artifacts signal the handoff point to the perturbation engine*

---

### Render gallery

![psych spine](screenshots/05_psych_spine.png)
*PSYCH · 4,398,046,511,104× zoom · 8,192 iterations · 3.58s*

![seahorse spirals](screenshots/06_seahorse_spirals.png)
*PSYCH · 536,870,912× zoom · 768 iterations · 0.68s — Seahorse Valley*

![ultra spiral](screenshots/07_ultra_spiral.png)
*ULTRA · 35,184,372,088,832× zoom · 1,000,000 iterations · 1.89s*

![hybrid corona](screenshots/08_hybrid_corona.png)
*HYBRID · 35,184,372,088,832× zoom · 1,000,000 iterations · 1.95s*

---

## Installation

```bash
git clone https://github.com/yourusername/fractal-explorer
cd fractal-explorer
pip install -r requirements.txt
python fractal_explorer.py
```

### Requirements

```
numpy
Pillow
blessed
numba          # optional — CPU JIT, significant speedup
mlx            # optional — Apple Silicon GPU (macOS only)
mpmath         # required for deep zoom (perturbation theory)
```

---

## Controls

| Key | Action |
|-----|--------|
| `arrows` | Pan |
| `z / x` | Zoom in / out |
| `p` | Cycle palette |
| `m` | Cycle fractal mode (Mandelbrot / Burning Ship / Tricorn) |
| `j` | Toggle Julia mode |
| `o` | Trace orbit at viewport center |
| `[ / ]` | Iterations down / up |
| `1–9` | Jump to preset location |
| `r` | Full-resolution iTerm2 render |
| `S` | Save PNG to Desktop |
| `q` | Quit |

### Presets

| Key | Location |
|-----|----------|
| `1` | Seahorse Valley |
| `2` | Elephant Valley |
| `3` | Double Spiral |
| `4` | Tendrils |
| `5` | Mini Brot |
| `6` | Lightning |
| `7` | Misiurewicz ∞ |
| `8` | Needle ∞ |
| `9` | Seahorse ∞ |

---

## Palettes

`HYBRID` · `RED CRT` · `AMBER` · `FIRE` · `ICE` · `ULTRA` · `PSYCH` · `GRAY`

---

## ⚠️ Memory warning

**High iteration counts at low zoom depths will exhaust system memory** and may require a hard power cycle to recover.

The engine tiers have very different memory profiles:

- **MLX / numba** (shallow zoom) — compute every pixel independently in parallel. Memory scales with `iterations × pixels × threads`. Above ~25k iterations at default zoom you are in dangerous territory on 16GB systems.
- **Perturbation engine** (zoom > 1e13×) — compute one reference orbit, cheap delta math per pixel. Memory footprint stays flat regardless of iteration count. Deep zoom is actually *safer* than shallow zoom at high iterations.

The explorer will warn you in the status bar when you push iterations into the risk zone at low zoom depths. Heed the warning.

---

## How perturbation theory works

At extreme zoom the viewport window in the complex plane becomes vanishingly small. Standard float64 arithmetic loses the ability to distinguish adjacent pixels — they all round to the same coordinate.

The fix: instead of computing absolute coordinates for each pixel, compute *offsets* (δC) from a single reference point. The reference orbit is computed once at arbitrary precision using mpmath. Each pixel then runs a cheap perturbation loop:

```
δZ_{n+1} = 2·Z_ref[n]·δZ_n + δZ_n² + δC
```

This keeps all the sensitive arithmetic in one high-precision computation while the per-pixel work stays in fast float64.

When the reference orbit degenerates (glitch detection), the engine automatically selects new reference points and re-renders affected pixels. Up to 8 correction passes per frame.

---

## Performance

All benchmarks on MacBook Pro (Apple Silicon), iTerm2, 2996×1810 output resolution, 100,000 iterations.

| Zoom depth | Engine | Render time |
|------------|--------|-------------|
| 256× | MLX GPU | 118s* |
| 4T× | CPU perturbation | 1.41s |
| 35T× | CPU perturbation | ~1.9s |
| 70T× | CPU perturbation | ~1.9s |
| 140T× | CPU perturbation | 6.6s |

*MLX at shallow zoom is doing full per-pixel GPU computation at 100k iterations — expected. Perturbation theory inverts this: deeper zoom, faster render.

We haven't found the iteration ceiling yet. 100k iterations at 140T× in under 7 seconds. The methodology has headroom.

---

## Fractal modes

- **Mandelbrot** — the classic. Full perturbation engine support for extreme zoom.
- **Burning Ship** — uses `|Re(z)| + i|Im(z)|` instead of `z`. Different boundary geometry, same engine.
- **Tricorn** — conjugate Mandelbrot variant. Reflection symmetry along real axis.
- **Julia sets** — toggle with `j`. Any viewport center becomes the Julia parameter.

---

## Requirements file

```
numpy>=1.24
Pillow>=9.0
blessed>=1.19
mpmath>=1.3
numba>=0.57     # optional
mlx>=0.5        # optional, Apple Silicon only
```

---

## License

MIT
