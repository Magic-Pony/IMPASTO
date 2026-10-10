# Stage 3 asymmetric overpaint objective: controlled trial

This trial starts after local commit `289b784` and reuses the existing
`outputs/stage2_soft_edge/impasto_unet_best.pt` checkpoint. It does not collect
self-play data or retrain the forward model. The only planner change is a
twofold penalty for predicted grayscale darker than the target. The default
planner still uses symmetric L1. H=5, execute 2 strokes per replan, and the
published Fig. 8 panel targets are unchanged.

Command:

```bash
IMPASTO_STAGE3_CONFIG=configs/stage3_refined.json bash scripts/run.sh scripts/run_stage3.py --report-dir reports/stage3_overpaint_2p0 --overpaint-weight 2.0
bash scripts/run.sh scripts/audit_refinement.py --report-dir reports/stage3_overpaint_2p0 --out-dir reports/refinement_overpaint_2p0
```

| Task | Prior best weighted gray L1 | Trial weighted gray L1 | Prior best foreground IoU | Trial foreground IoU | Trial recall | Trial precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flower | 0.01486 | 0.01539 | 0.758 | 0.776 | 0.797 | 0.967 |
| Fish | 0.01375 | 0.01543 | 0.715 | 0.666 | 0.694 | 0.942 |

The comparison is descriptive, not a one-variable ablation against the prior
best: attempt 6 used the earlier refined checkpoint and different target
prefix/opacity settings. The new trial's resolved settings and model checksum
are in `../stage3_overpaint_2p0/resolved_config.json` and `metrics.json`.

Visual inspection of [Flower round 7](flower_audit_07.png) shows a more
restrained vertical stem. In the [full Flower progression](flower_rounds_montage.png),
the flower remains recognizable but flat: the simulator deposits crisp opaque
swaths where the source has soft mixed edges. The [Fish comparison](fish_comparison.png)
shows a narrower central body but worse correspondence of the separated head,
body and fins. The Fish recall and IoU decline confirms the visual result.

This objective change is **not recommended** as the default. It cannot by
itself correct candidate geometry or reproduce low-resolution target
appearance. The experiment remains available as a switchable local extension
for later work. See the [generated stage-3 report](../stage3_overpaint_2p0/report.md),
[Flower video](../stage3_overpaint_2p0/flower_painting.mp4), and
[Fish video](../stage3_overpaint_2p0/fish_painting.mp4) for complete execution.
