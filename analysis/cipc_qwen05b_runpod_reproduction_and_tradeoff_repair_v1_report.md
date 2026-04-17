# Qwen 0.5B Runpod Reproduction And Trade-Off Repair V1

## Environment

- preflight: passed
- torch: `2.4.1+cu124`
- gpu: `NVIDIA H100 80GB HBM3`
- tracked Belief-R commitment-control assets: present

## Server-Side Runs

| run | overall answer | overturn answer | no-overturn answer |
|---|---:|---:|---:|
| local control_focused_v1 | 0.7923 | 0.7670 | 0.8889 |
| local highrank_v1 | 0.8462 | 0.8641 | 0.7778 |
| Runpod control_focused_v1 reproduction | 0.8154 | 0.8932 | 0.5185 |
| Runpod highrank_v1 reproduction | 0.8538 | 0.9029 | 0.6667 |
| Runpod tradeoff_repair_v1 | 0.8538 | 0.9320 | 0.5556 |
| NumPy CIPC baseline | 0.7846 | 0.7961 | 0.9259 |
| frozen prompt baseline | 0.3615 | 0.2718 | 0.5926 |

## Main Readout

The Runpod environment does not reproduce the local frontier shape.

- `control_focused_v1` is no longer the more balanced point.
- both Runpod reproductions are much more aggressive on overturn than the local
  report.
- the maintain-side collapse is much worse than the local report, especially
  for the reproduced `control_focused_v1`.

The single-variable repair run moved in the wrong direction for the current
server regime.

- change tested:
  keep `highrank_v1` capacity and schedule, reduce `answer_loss_weight`
  from `1.0` to `0.7`
- result:
  overturn improved from `0.9029` to `0.9320`
- but no-overturn fell from `0.6667` to `0.5556`
- overall stayed flat at `0.8538`

So the tested repair did not repair the trade-off. It pushed the model further
toward replacement.

## Recommendation

Use the Runpod `highrank_v1` reproduction as the current frontier point on this
server.

Next highest-value run:

- keep the Runpod `highrank_v1` config fixed
- move in the opposite direction from `tradeoff_repair_v1`
- test a more conservative answer objective, starting with
  `answer_loss_weight > 1.0`
- do not add new axes in the same run

This is the cleanest next step because the only tested repair lever so far
already showed the sign of effect:

- lower answer-loss weight increases overturn
- lower answer-loss weight hurts no-overturn

The next repair should therefore test the opposite sign while preserving the
same split, seed, model family, LoRA rank, and epoch count.
