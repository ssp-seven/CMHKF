# CMHKF · ACL 2025

**Cross-Modality Heterogeneous Knowledge Fusion for Weakly Supervised Video Anomaly Detection**

[![ACL 2025](https://img.shields.io/badge/ACL-2025-blue)](https://aclanthology.org/2025.acl-long.1524/) [![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](https://aclanthology.org/2025.acl-long.1524.pdf) [![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

CMHKF is a framework for weakly supervised video anomaly detection. It combines video, audio, and text information to detect and localize anomalous events using video-level labels during training.

![CMHKF framework](docs/images/cmhkf-framework.png)

*Framework overview from Figure 1 of the [ACL 2025 paper](https://aclanthology.org/2025.acl-long.1524/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

The framework uses cross-modality video-text knowledge alignment (CVKA), audio modality feature adaptive extraction (AFAE), and multi-modality knowledge adaptive fusion (MKAF).

## Installation

The commands below use Python 3.8 and CUDA 11.3 on Linux. A CUDA-enabled NVIDIA GPU is required. For other CUDA versions, consult the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

```bash
git clone https://github.com/ssp-seven/CMHKF.git
cd CMHKF
conda create -n cmhkf python=3.8 -y
conda activate cmhkf

python -m pip install setuptools==68.2.2
python -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install numpy==1.23.5 pandas==1.5.3 scipy==1.10.1 \
  scikit-learn==1.3.2 ftfy==6.2.0 regex==2024.5.15 \
  tqdm==4.66.4 Pillow==9.3.0
```

The model uses the CLIP implementation in `src/clip/`. The first run downloads the CLIP ViT-B/16 weights to `~/.cache/clip/`. The supplied `src/requirements.txt` records the original environment and includes machine-specific paths; use the commands above for a portable setup.

## Data preparation

Download [XD-Violence](https://roc-ng.github.io/XD-Violence/) and prepare temporally aligned `float32` NumPy features for each video:

- CLIP ViT-B/16 video features with shape `[T, 512]`.
- VGGish audio features with shape `[T, 128]`.

Here `T` is the number of time steps. Evaluation expands each time step into 16 frames, so align the features and annotations accordingly. The repository does not provide a complete batch feature-extraction pipeline.

Update the `path` column in the following CSV files to point to your local features:

| Split | Video features | Audio features |
| --- | --- | --- |
| Train | `list/xd_CLIP_rgb.csv` | `list/xd_audio.csv` |
| Test | `list/xd_CLIP_rgbtest.csv` | `list/xd_audio_test.csv` |

Each CSV uses a `path,label` header. Replace only the `path` values with absolute paths to your feature files. Keep the video and audio rows paired and preserve the test order. The training list contains multiple visual crops per video with the corresponding audio path repeated.

The default test list uses the ground truth already in `list/gt.npy`, `list/gt_segment.npy`, and `list/gt_label.npy`. Use test features that match the annotations for each video. The total number of video feature steps multiplied by 16 must equal the length of `list/gt.npy`.

## Training

From the repository root, run:

```bash
cd src
mkdir -p ../model
CUDA_VISIBLE_DEVICES=0 python train.py
```

The training script evaluates the model periodically and saves weights to `model/model.pth` when average precision improves. See `src/option.py` for the default paths and options.

## Evaluation

After training has saved `model/model.pth`, run from `src/`:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```

The script reports frame-level detection and temporal localization metrics.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{wang-etal-2025-cmhkf,
  title = "{CMHKF}: Cross-Modality Heterogeneous Knowledge Fusion for Weakly Supervised Video Anomaly Detection",
  author = "Wang, Guohua and Song, Shengping and He, Wuchun and Zheng, Yongsen",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2025",
  address = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  pages = "31594--31607",
  url = "https://aclanthology.org/2025.acl-long.1524/",
  doi = "10.18653/v1/2025.acl-long.1524"
}
```

## Acknowledgments and license

This project builds on [CLIP](https://github.com/openai/CLIP), [VadCLIP](https://github.com/nwpu-zxr/VadCLIP), and [XD-Violence](https://roc-ng.github.io/XD-Violence/). The code is released under the [Apache License 2.0](LICENSE). Dataset and pretrained model terms apply separately.
