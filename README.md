# Upscale Pro 4K

A smart, self-improving AI video upscaler that runs free on your own PC (any Vulkan GPU, including this PC's AMD RX 6700 XT).
It doesn't apply one recipe to every video. It **analyses** the source, **A/B tests** several upscaling pipelines on the video's own hardest scenes, renders the **winner**, and **learns** from every run.

How it works in depth: **[DESIGN.md](DESIGN.md)**.

## Use

### Web Studio (Recommended)
Double-click **`start_web.bat`**, or run:
```bash
py -3.11 web_server.py
```
Then navigate to: **`http://127.0.0.1:8000`** in your browser.

### Command Line
Drag a video onto **`Upscale.bat`**, or run:

```bash
py -3.11 upscale.py "C:\path\video.mp4"
```

Here's what happens:

1. **New-tech check** (once a day). It looks for newly published models and engine builds. New models join as *trial* challengers.
2. **Analysis.** It measures compression, noise, film grain, *true* detail resolution ("fake HD" detection), banding, interlacing/telecine, black bars, HDR, faces and content type (anime or live action). It prints what it will **fix** and what it will **preserve**.
3. **Plan.** It shortlists models suited to *this* source, using what has won before on similar videos.
4. **A/B tournament.** Every candidate upscales the same sample crops. A judge scores fidelity, detail, texture (anti-plastic), flicker, blocking and halos. Finishing touches are then tuned (blend-back, sharpening), and grain, debanding and deflicker are applied where the source needs them.
5. **4K or Full HD.** It only offers Full HD as a last resort, when 4K would mean inventing most of the picture or would look no better. In that case it tells you why and asks.
6. **Render** the full video with the winning recipe. It's resumable: re-run the same command after a crash.
7. **Learn.** Scores are saved per kind of source. Trial models get promoted or retired. Speeds are measured for accurate ETAs.

Outputs:
- `video_4K.mp4` (or `video_FHD.mp4`) next to the source
- a run report in `data/runs/<run>/report.md`
- a visual contact sheet of the candidates in `tournament_sheet.png`

## Commands

Diagnosis and plan only, no AI work:

```bash
py -3.11 upscale.py analyze video.mp4
```

Analysis plus A/B tournament, without the long render:

```bash
py -3.11 upscale.py video.mp4 --plan-only
```

Upscale a 10-second sample and write a plain-vs-AI split-screen:

```bash
py -3.11 upscale.py video.mp4 --start 60 --duration 10 --compare
```

Check for new models and engine builds now:

```bash
py -3.11 upscale.py update
```

List models with their status, wins and measured speed, plus the research watchlist:

```bash
py -3.11 upscale.py models
```

List past runs:

```bash
py -3.11 upscale.py runs
```

Teach the judge your taste:

```bash
py -3.11 upscale.py feedback <run-id> --prefer C2 --rating 4
```

### Useful options

| Option | Effect |
|---|---|
| `--fallback ask/auto/never` | What happens when Full HD is recommended. The default `ask` stops before rendering if it can't ask. |
| `--candidates 3 --trials 1` | How many incumbent models and new trial models compete. |
| `--samples 4` | Number of sample scenes in the tournament. |
| `--model NAME` | Force a model (see `models`). |
| `--allow-nc` | Allow non-commercial-licence models (UltraSharp, Remacri, UltraMix). |
| `--quick` | Skip the tournament and use the planner's top pick. |
| `--max-days 2` | Time budget. Slower models are de-prioritised if they won't fit. |
| `--target 4k/dci4k/1440p/1080p/WxH`, `--mode fill`, `--pad` | Output size and shape. |
| `--codec x265/av1/amf`, `--crf 16` | Final encode. The default is 10-bit HEVC, CRF 16, preset slow. |
| `--offline`, `--no-engine-update` | Disable the internet check, or engine auto-adoption. |

## Setup (already done on this PC)

```bash
py -3.11 -m pip install -r requirements.txt
```

```bash
py -3.11 setup_tools.py
```

FFmpeg must be on PATH. The `upscayl-bin` engine is in `tools/upscayl/`. Models download on first use into `models/`.

Optional: `pip install pyiqa` adds a learned perceptual quality metric (TOPIQ-NR) to the judge. It needs PyTorch, which is about 1 GB.

## Speed (RX 6700 XT, measured)

| Model class | Seconds per megapixel of input | 540p source (0.3 MP) |
|---|---|---|
| Fast (Compact/SRVGG: WDN, LSDIRCompactC3, animevideov3) | ~0.3–1.8 | ~0.1–0.5 s per frame |
| Large (ESRGAN: Nomos8kSC, NMKD, x4plus, LSDIRplusC) | ~5.6–8.5 | ~2–2.5 s per frame |

The tournament itself takes a few minutes. The full render ETA is shown before it starts and after every chunk.
