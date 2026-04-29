# VM Selection Notes

Checked on 2026-04-29 KST for the `nemo-488500` GCP project and the configured AWS account.

## Priority

1. A100 on-demand if immediately available.
2. Existing GCP L4 VM fallback.

## AWS

- Region: `ap-northeast-2`
- A100 instance offering exists for `p4d.24xlarge` in `ap-northeast-2b` and `ap-northeast-2d`.
- Blocking quota: `Running On-Demand P instances = 0`.

Conclusion: AWS A100 on-demand cannot be launched immediately from the current quota state.

## GCP

- Project: `nemo-488500`
- A2 machine types and A100 accelerator zones exist.
- Blocking quota: `NVIDIA_A100_GPUS = 0` and `NVIDIA_A100_80GB_GPUS = 0` in checked regions.
- Some regions expose preemptible A100 quota, but that is not the requested on-demand path.
- `NVIDIA_L4_GPUS = 1` is available in checked regions.

Existing useful fallback candidates:

```bash
gcloud compute instances start pxr-chemprop-l4-image-run \
  --project nemo-488500 \
  --zone us-central1-a
```

`pxr-chemprop-l4-image-run` is a STANDARD `g2-standard-4` VM with one NVIDIA L4 and a CUDA 12.9 deep learning image.

`cuda-l4-dev-lesson09` is also a STANDARD `g2-standard-4` L4 VM in `us-east4-c`, but it hit an L4 stockout on 2026-04-29.
