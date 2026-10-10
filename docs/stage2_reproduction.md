# Stage 2: Pixel Dynamics Model

## Paper specification

The model predicts the complete next grayscale canvas patch from the current patch and a rasterized candidate stroke. The image input is `100 x 100 x 2`: current grayscale observation and pre-rendered grayscale stroke. The output is the complete next `100 x 100 x 1` observation, with a sigmoid output. The action encoder receives six values; the paper's architecture description lists `(x0, y0, length, bend, force, angle)`. An MLP embedding is fused into the 64-channel encoder level. The U-Net feature widths shown in Appendix B are `32, 64, 128, 256`; encoder blocks use Conv2d, BatchNorm, LeakyReLU, and dropout, while decoder blocks upsample, convolve, normalize, and apply LeakyReLU. The output head uses two convolutions and sigmoid.

Training uses the paper's weighted L1 objective, with weights derived from a dilated difference mask between consecutive observations. The appendix specifies Adam with learning rate `5e-4`, weight decay `1e-3`, batch size `120`, `1000` epochs, and a `0.75` learning-rate multiplier every 100 epochs. Paper Fig. 5 labels `900` as the number of training samples and the text specifies validation as 10% of the total data. This reproduction collects 900 training and 100 held-out validation samples, grouped by episode to prevent adjacent canvas states crossing the split.

## Local simulation contract

Each transition is captured from the live Stage-1 `Canvas` before and after a completed simulated stroke. No synthetic target image is substituted for the simulator observation. Actions are sampled uniformly from the existing verified Stage-1 sampler bounds: `x0 in [-0.11, 0.04] m`, `y0 in [-0.11, 0.11] m`, `length in [0.025, 0.085] m`, `bend in [-0.035, 0.035] m`, `angle in [-pi, pi]`, and target force in `[0.4, 1.5] N`. These task bounds are not specified numerically in the paper. The paper's stated `0.1-4 N` is the F/T sensor measurement range, not a published action sampling interval.

The original `configs/stage2.json` uses a 100-square crop centered on the sampled stroke bounds with a 25 mm local margin, metric resampling, and a four-color subset cycled every 25 transitions. The action channel uses disk stamps along the quadratic Bezier curve; it encodes candidate geometry, while the simulator's observed next frame supplies the deposition response. Rendered color is converted to grayscale and remains outside the six-dimensional geometry action.

The later `configs/stage2_refined.json` experiment uses an 18 mm margin, 24 colors cycled each transition, shorter strokes (4--105 mm), and the force-to-radius relation `r = 0.4 mm + 5.8 mm * F^1.65`. The soft-edge experiment changes only the observed deposition profile to a Gaussian with a smooth finite-support taper; the disk-stamp action rasterizer remains unchanged. Its data and model outputs are isolated under `stage2_soft_edge`. This is an unvalidated local simulation hypothesis, not a correction to the paper. See `reports/stage2_soft_edge/status.md` for the intermediate submission status.

The paper does not publish numeric action bounds, action normalization, crop margin, footprint calibration coefficients, dropout rate, or the scalar mapping from the dilated mask to loss weights. Those values are exposed in `configs/stage2.json` and reported as local choices. The signed local bend is mapped symmetrically around `0.5` so its sign is retained. The validation score is compared with the paper's unweighted L1 result of `0.0285`; the local contact-driven paint model is not the real robot, so this value is a target for comparison rather than a guaranteed equivalence.

## Dataset and training

The collector creates ten deterministic episodes of 100 successful transitions. Episodes 0-8 form the 900-sample training set; episode 9 is the 100-sample validation set. Failed attempts are recorded in the collection manifest and replaced up to the configured attempt limit. Dataset arrays contain the two input channels, next-state target, normalized local action, episode and step IDs, color label, crop transform, force result, and collection seed.

The signed bend normalization is centered at 0.5; a data audit caught and corrected an earlier encoding that clipped negative bend values. Raw action geometry and simulator observations were retained, and the stored normalized action arrays were rebuilt from those raw values. The final action encoding preserves positive and negative bend signs across all 1,000 transitions.

Run the collection, training, and evaluation from the repository root:

```bash
bash scripts/run.sh scripts/collect_selfplay.py
bash scripts/run.sh scripts/train_dynamics.py
bash scripts/run.sh scripts/evaluate_dynamics.py
```

The collection manifest, training curves, validation metrics, model checksum, and representative predictions are written under `reports/stage2/`. Raw samples and checkpoints are kept under ignored `data/` and `outputs/` directories; their generation configuration and aggregate results are versioned with the report.

## Completed run

The run collected 1,000 successful transitions in 179.9 s using four CPU workers. Mean contact fraction was 0.99988; mean force RMSE was 0.0310 N; mean trajectory RMSE was 2.59 mm. Training used the RTX 4090 D for 1,000 epochs and took 593.1 s. The best checkpoint was selected at epoch 176. On the 100 episode-held-out validation transitions, full-patch unweighted L1 was 0.00977, changed-region L1 was 0.05066, and background L1 was 0.00668. The full-patch value is below the paper's reported 0.0285 reference, though the local simulator and real-robot data are not identical evaluation domains.

The report includes the loss curves, six next-frame prediction examples, measured simulation quality, exact hyperparameters, and checkpoint checksum. The validation episode is held out from gradient updates but is used for checkpoint selection; it is not a separate untouched test set. The model predicts individual next frames only in Stage 2; multi-step MPC rollout and the painting tasks belong to Stage 3.
