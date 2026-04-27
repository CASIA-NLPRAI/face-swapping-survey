# FaceSwap Evaluation

A multi-metric evaluation framework for face swapping results.

## Metrics

| Metric | File | Description |
|--------|------|-------------|
| ID Retrieval | `eval_id_retrieval.py` | Top-1 identity retrieval accuracy via face recognition |
| ID Similarity | `eval_id_similarity.py` | Cosine similarity between source and swapped face features |
| Pose Error | `eval_pose_err.py` | L2 distance of head pose angles (yaw/pitch/roll) via Hopenet |
| Expression (3DDFA) | `eval_exp_3ddfa.py` | L2 distance of expression parameters via 3DDFA (MobileNet) |
| Expression (FWH) | `eval_exp_facewarehouse.py` | L2 distance of expression parameters via FaceWarehouse (ResNet50) |
| FID | `eval_FID.py` | Frechet Inception Distance between original and swapped face distributions |
| SSIM | `eval_SSIM.py` | Structural Similarity between source and swapped faces |

## Project Structure

```
eval_faceswap/
├── main.py                  # Entry point
├── config.py                # Path and device configuration
├── utils.py                 # Common utilities (name parsing, landmark reading, etc.)
├── eval_id_retrieval.py     # ID Retrieval metric
├── eval_id_similarity.py    # ID Similarity metric
├── eval_pose_err.py         # Pose Error metric
├── eval_exp_3ddfa.py        # Expression Error (3DDFA)
├── eval_exp_facewarehouse.py# Expression Error (FaceWarehouse)
├── eval_FID.py              # FID score
├── eval_SSIM.py             # SSIM metric
├── prepare_id_features.py   # Pre-compute identity features for retrieval
├── anno.csv                 # Annotation file for identity feature preparation
├── DDFA/                    # 3DDFA model for expression estimation
│   ├── DDFA.py
│   └── models/
│       └── mobilenet_v1.py
├── face_recognition/        # SphereFace + CosFace for identity verification
│   ├── recognition.py
│   ├── net.py
│   ├── face_align.py
│   └── utils.py
├── facewarehouse/           # FaceWarehouse model for expression estimation
│   ├── exp_estimate.py
│   ├── load_data.py
│   ├── preprocess_img.py
│   ├── BFM/                 # 3D Morphable Model data
│   ├── lib_py/              # Utility libraries
│   └── network/
│       └── resnet50_task.py
├── pose_estimation/         # Hopenet for head pose estimation
│   ├── pose_estimate.py
│   └── hopenet.py
├── pytorch_fid_new/         # FID calculation (based on pytorch-fid)
│   ├── fid_score.py
│   └── inception.py
└── id_prepare/              # Pre-computed identity features storage
```

## Requirements

- Python 3.7+
- PyTorch >= 1.6
- torchvision
- OpenCV (`opencv-python`)
- numpy
- scipy
- Pillow
- tqdm
- pytorch-msssim

## Pre-trained Models

Download the following model weights and place them in the corresponding directories:

| Model | Path | Source |
|-------|------|--------|
| CosFace | `face_recognition/CosFace_ACC99.28.pth` | SphereFace with CosFace weights |
| 3DDFA | `DDFA/models/phase1_wpdc_vdc.pth.tar` | [3DDFA](https://github.com/cleardusk/3DDFA) |
| Hopenet | `pose_estimation/hopenet_robust_alpha1.pkl` | [Hopenet](https://github.com/natanielruiz/deep-head-pose) |
| FaceWarehouse | `facewarehouse/network/th_model_params.pth` | FaceWarehouse 3D face model |
| FID Inception | `pt_inception-2015-12-05-6726825d.pth` | [pytorch-fid](https://github.com/mseitzer/pytorch-fid) |
| BFM | `facewarehouse/BFM/mSEmTFK68etc.chj` | 3D Morphable Model |
| BFM Landmarks | `facewarehouse/BFM/similarity_Lm3D_all.mat` | Standard 3D landmarks |

## Data Organization

Expected directory layout:

```
original_data/
├── faces/                    # Original face images: {id}-{idx}.png
│   ├── 0001-0_0.png
│   └── ...
└── faces_landmarks/          # 5-point landmarks: {id}-{idx}.txt
    ├── 0001-0_0.txt
    └── ...

swap_results/
└── {method_name}/
    └── {swap_type}/          # Swapped face images: {src}-{tgt}.png
        ├── 0001-0_0-0619-0_0.png
        └── ...
```

Swapped image naming convention: `{src_id}-{src_idx}-{tgt_id}-{tgt_idx}.png`

## Usage

1. Modify paths in `config.py` or set environment variables:

```bash
export EVAL_RESULT_ROOT=/path/to/swap/results/
export EVAL_ORI_DATA_ROOT=/path/to/original/faces/
```

2. (Optional) Pre-compute identity features for ID Retrieval:

```bash
python prepare_id_features.py
```

3. Run evaluation:

```bash
# Evaluate specific methods
python main.py --methods SimSwap FaceShifter --gpu 0

# Evaluate with custom swap types
python main.py --methods SimSwap --types all_simple all_cross_ethnicity --gpu 0
```

Results are saved to `eval_results_{method}_{type}.txt` in the result root directory.
