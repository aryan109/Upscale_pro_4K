# Upscale Pro 4K: how the smart engine works

There's no single upscaling recipe that suits every video. A 360p Pinterest re-upload, a grainy film scan, an interlaced DVD and a clean anime episode each need different treatment.
The engine therefore treats every job as an experiment: it measures the source, proposes several pipelines, tests them against each other on the video's own hardest scenes, and renders the winner.
It also remembers what won, so the next similar video starts from evidence rather than guesswork.

```
 ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────┐
 │ new-tech │──►│ analyse  │──►│ plan          │──►│ A/B          │──►│ 4K or    │──►│ render  │
 │ check    │   │ source   │   │ candidates    │   │ tournament   │   │ Full HD? │   │ + learn │
 └──────────┘   └──────────┘   └──────────────┘   └──────────────┘   └──────────┘   └─────────┘
   registry.py    analyze.py     planner.py         tournament.py      upscale.py     render.py
                                 knowledge.py ◄──── judge.py ──────────────────────► knowledge.py
```

## 1. Analysis (`engine/analyze.py`)

The analyser decodes 4 to 12 evenly spaced sample points (3 consecutive frames each), plus one 40-second pass through FFmpeg's detection filters.

| Measurement | How | Used for |
|---|---|---|
| Interlacing / telecine | `idet` (field-order counts, repeated fields) | `bwdif` deinterlace, or inverse telecine (`fieldmatch,decimate`) back to 23.976 fps |
| Black bars | `cropdetect` (accumulated) | Crop before the AI, so it doesn't waste work on bars or smear their edges |
| Compression | Own 8x8 grid-step ratio, `blockdetect`, and bits per pixel normalised to H.264 (HEVC 0.65x, AV1 0.55x, MPEG-2 2x…) | Compression-trained models, deblock variants |
| Noise | Immerkær estimator on low-gradient pixels | Denoise variants, model choice |
| Grain | Temporal noise in static areas (random in time = real grain, not texture) | Re-grain after the AI, to *preserve* the look |
| **True detail resolution** | Denoise, then shrink by f and restore: the largest f that stays above 39 dB is detail the file never held | Spots "fake HD" (e.g. 1080p holding 540p of detail). The AI is fed the picture at its true detail level, the scale its training expects |
| Banding | Gentle, texture-free gradients that sit flat then step by exactly 1 level | `deband` + 10-bit output |
| Content type | Flat-fill share, line-edge density, natural-texture share, faces | Anime vs live-action model families. If ambiguous, both compete |
| Faces | Haar cascade | Small faces raise the hallucination risk, so fidelity is favoured |
| Motion, darkness, texture | Frame differences, luma statistics, local variance | Notes; texture protection; sample-scene selection |

The output is a **profile**: a category key such as `live/sd/moderate/clean`, plus lists of what to **fix** and what to **preserve**.

**4K feasibility** is the enlargement factor from the true detail level to 2160 lines:

- up to 4.2x: low risk
- up to 6x: medium
- up to 9x: high
- above 9x: extreme

Heavy compression or high noise raises the risk by one level.

## 2. Model registry and self-update (`engine/registry.py`)

The engine is `upscayl-bin`, the maintained fork of Real-ESRGAN-ncnn-vulkan. It runs on any Vulkan GPU, including this PC's AMD card, and loads any NCNN model.

The seed catalogue was researched in October 2026 from OpenModelDB and the Upscayl and Real-ESRGAN repositories. Each model is tagged with:

- its domain (live or anime)
- the degradations it was trained on (jpeg, noise, blur, clean)
- its speed class
- its licence

Non-commercial models (CC-BY-NC-SA) are excluded unless you pass `--allow-nc`.

**Before every run** (at most once every 24 hours; force it with `upscale.py update`), the engine does three things:

1. **New models:** it lists the official Upscayl model repositories. Any new `.param`/`.bin` pair is added as a **trial**, with its metadata guessed from the name and size.
2. **New engine:** if `upscayl-ncnn` has a newer release, it downloads it into its own folder and runs an **engine A/B test** against the current engine. If the output is identical (PSNR of at least 40 dB) and it runs correctly, it's adopted, and the previous path is kept for rollback. If not, it's rejected and logged.
3. **Research watch:** it tracks new models published on OpenModelDB. Most new models are PyTorch-only and can't run on the Vulkan engine yet, so they go on a watchlist (see `upscale.py models`). Any with an NCNN build are flagged.

Only allow-listed official GitHub sources are contacted. Models are weight files, not code.

## 3. Planner (`engine/planner.py`)

The planner ranks every available model for this profile using two inputs:

- **Rules:** compression-trained models get a bonus on compressed sources. Clean-source specialists get a penalty there. Denoisers get a bonus on noisy sources. Strong denoisers get a penalty where grain or texture must be kept. Speed counts against a `--max-days` budget.
- **Learned history:** each model's average tournament score in the most specific matching category: `live/sd/moderate/clean`, then `live/sd/moderate`, then `live/sd`, then `live`.

It then shortlists:

- the top *N* active models, from both families when the content is mixed
- the current **champion** for this category, which always defends its title
- the best-matching **trial** models (new tech challenging the incumbents)
- **pre-processing variants** of the leader when needed: deblock-first for heavy blocking, NL-means-first for real noise

## 4. A/B tournament (`engine/tournament.py`) and judge (`engine/judge.py`)

1. The tournament picks the most detailed sample scenes (plus a dark one), and in each the most textured crop. That's where upscalers differ most.
2. Every candidate renders the same crops (3 consecutive frames, after 2 warm-up frames for temporal filters).
3. **Stage 2** reuses the two best AI outputs and sweeps the finishing steps. Blend (0, 15 or 30% of the plain Lanczos picture mixed back in, to recover natural texture) and CAS sharpening cost no extra AI time.
4. The **judge** scores each result (0–100) on fidelity, detail gain, texture retention (anti-plastic), temporal stability, leftover blocking, ringing/halos and colour drift. If `pyiqa` is installed, a learned perceptual metric (TOPIQ-NR) is added.
5. A contact sheet (`data/runs/<id>/tournament_sheet.png`) shows the candidates side by side, so a human can check the judge.
6. **Finishing rules (preserve):**
   - Re-grain to the measured source grain level.
   - Add light grain if the winner still looks waxy.
   - Deband when banding or dark scenes are present.
   - Deflicker if the winner shimmers more than the source.

## 5. 4K or Full HD (`upscale.py`)

Full HD is a **last resort**. It's recommended only when at least one of these is true:

- the risk is extreme (4K would mean inventing most of the picture),
- the risk is high *and* the tournament shows a Full-HD master, viewed on a 4K screen, scoring within 1 point of the 4K master,
- the best 4K candidate scores below the quality floor.

The user is then told why and asked to choose. `--fallback auto` picks Full HD, `--fallback never` forces 4K. A non-interactive run stops before the long render, so the person decides.

## 6. Learning (`engine/knowledge.py`)

- Every tournament result is stored per category scope. Later runs on similar footage start with proven strategies (the planner priors), so the system gets better at guessing first time.
- **Trial lifecycle:** a trial model is *promoted* when it wins, or matches the incumbents within 1.5 points over 2 tournaments. It's *retired* when it loses by more than 5 points in 4 consecutive trials.
- **Speed learning:** measured seconds per megapixel per model replaces the speed priors, so ETAs and time budgets become accurate.
- **Human feedback:** run `upscale.py feedback <run> --prefer C2 --rating 4`. If you prefer a candidate the judge ranked lower, the judge's metric weights shift toward the metrics where your choice was better. Over time the automatic judge learns your taste. Ratings also adjust the stored score of the strategy that was used.

## What "self-improving" honestly means here

The system can't invent new AI models. What it does automatically:

- notices when new models or engine builds are published,
- tests them on your real footage against the current best,
- adopts whatever measurably wins and retires what loses,
- tunes its own recipe choices and judging toward results that you confirm look better.

When the field produces a better model (e.g. a new compression-aware one), it enters as a trial on the next run and has to *earn* its place.

## Known limits

- Without a true 4K original, the judge uses no-reference proxies. They're good at catching plastic, ringing, flicker and drift, but they're not human eyes. That's why there's a contact sheet and a feedback command.
- New PyTorch-only research models (often the strongest ones) need conversion to NCNN before they can run on this AMD GPU. The watchlist flags them.
- Very small faces in low-res video are the hardest case for every model. The engine favours fidelity there instead of inventing features.
