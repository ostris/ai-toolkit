# Sigmoid timestep schedules in AI Toolkit

The schedule changes the sigmoid parameters over the course of FlowMatch training. In the job form, choose **Sigmoid** and open **Edit Timestep Schedule**. Add control points and apply the schedule to the draft. The same values appear in Advanced YAML under `train.timestep_sigmoid_schedule`.

Each control point has three fields.

| Field | Meaning |
| --- | --- |
| `start_step` | Absolute, zero-based Toolkit training step. Points must be in strictly increasing order. |
| `shift` | Finite normal offset. Negative values favor higher noise. |
| `scale` | Finite positive normal standard deviation. |

The sampler checks that both parameters fit its floating-point dtype before drawing noise. Sigmoid training uses a matching sigma table and accepts duplicate timesteps when looking up their noise levels.

Toolkit uses the exact values at each control point. Between points it linearly interpolates shift and scale over the training steps. Before the first point it uses `train.timestep_shift` and `train.timestep_scale`, which default to 0 and 1. After the last point it holds that point. Clear all points to use the scalar settings for the entire job.

The counter is the trainer's step index, in the same units as `train.steps`. Microbatches within one training step share its parameters. Resuming uses the restored step and the current job config. A schedule requires `noise_scheduler: flowmatch` and `timestep_type: sigmoid`. Linear overrides and models with forced training grids cannot use it.

## Visualizer

- Drag each point’s Start step, Shift, and Scale sliders to preview that point immediately. Start step sliders share one axis and stop before adjacent points. The fields accept exact values.
- The probability curve connects 200 five-timestep intervals. Its vertical scale covers the full interpolated schedule and the selected reference, and stays fixed while scrubbing and playing.
- Scrub or play the timeline to inspect the same interpolated settings that training uses at each step. A saved job's **Timestep Schedule** tab can follow its reported progress without editing that job.
- The custom curve uses Balanced rank selection. Choose one optional reference from the fixed sigmoid biases or a model's Shift distribution.
- For dynamic Shift, use an actual training bucket. The editor offers bucket dimensions from the job log when available and accepts exact dimensions in the draft. It derives the sequence length using the model's latent shape and the effective patch size used by Toolkit's trainer. Some video models also need the number of frames.

Toolkit's UI needs a rebuild and restart after applying the source patch. Start a fresh trainer process to load the Python changes.

## Developer checks

```sh
python -B -m unittest testing.test_sigmoid_parameters
cd ui
node --test test-sigmoid-schedule-*.cjs
```
