# Technical Project Report: Distracted Driver Detection Using Deep Learning

**Status:** dataset processed; a one-epoch custom-CNN run and one held-out driver evaluation completed. These are actual measured results for a deliberately short, low-resolution baseline, not a production-quality model or safety validation.

## Abstract

This project implements a modular image-classification pipeline for the State Farm distracted-driver dataset. Official driver metadata is used to isolate drivers across training, validation, and test partitions, reducing identity leakage. A custom convolutional classifier and MobileNetV2 transfer-learning option, CPU-conscious data loading, evaluation reports, and a demonstration web interface are provided. The downloaded labeled dataset was extracted and verified: 22,424 images across 26 drivers. A single epoch of custom-CNN training and one held-out evaluation have been measured; additional training and model comparisons remain pending.

## Introduction and problem definition

The task is to assign one of ten supplied in-cabin driver-image classes. It is a fixed-label image classification problem, not a real-time driver-monitoring, intent inference, or safety certification system. Similar images from one driver can share identity- and camera-specific cues, so random image-level evaluation can overstate generalization. This experiment groups by the provided driver ID.

## Objectives

1. Preserve the official ten-class mapping.
2. Validate image-to-driver mappings against `driver_imgs_list.csv`.
3. create reproducible, disjoint driver-wise train/validation/test partitions.
4. Implement a custom CNN and MobileNetV2 classifier option.
5. Provide streaming training, held-out evaluation, reports, tests, and an experimental web UI.

## Related work

**Literature review pending.** Add and verify primary literature and bibliographic metadata before citing. No unverified citation is asserted here.

## Dataset

Source: [State Farm Distracted Driver Detection competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection). The extracted labeled archive contains 22,424 images across 26 drivers. Required input includes the labeled class directories and official driver metadata CSV. The separate unlabeled competition test set is excluded from labeled evaluation.

| Code | Official label |
|---|---|
| c0 | Safe driving |
| c1 | Texting with right hand |
| c2 | Talking on phone with right hand |
| c3 | Texting with left hand |
| c4 | Talking on phone with left hand |
| c5 | Operating the radio |
| c6 | Drinking |
| c7 | Reaching behind |
| c8 | Hair and makeup |
| c9 | Talking to passenger |

## Preprocessing

Inputs are decoded as RGB, EXIF-oriented, resized to a configurable 160×160 resolution by default, converted to CHW float tensors, and linearly scaled to [-1, 1]. Images are streamed from disk. Training-only augmentation applies small rotations and mild brightness changes. Validation, test, and prediction inputs receive no augmentation.

## Driver-wise experimental design

The requested split is approximately 70/15/15 **by distinct drivers**, using fixed seed 48 and sequential group shuffling. The image counts need not match these proportions because per-driver image counts vary. Dataset preparation verifies all discovered files against official metadata and checks that the three driver ID sets are disjoint. Split CSVs and per-split image/driver/class counts are saved for reproducibility. If the required metadata is absent or cannot map files consistently, processing stops. Class presence in each split is data-dependent and is not artificially balanced.

## Architectures

### Custom CNN

Four double-convolution blocks (24, 48, 96, and 160 channels) use batch normalization, ReLU, max pooling, and spatial dropout. Adaptive global average pooling feeds a dropout and linear 10-logit head.

### MobileNetV2

Optional torchvision MobileNetV2, with a replacement 10-logit classifier and frozen feature backbone at initialization. ImageNet pretrained weights are an explicit option and require a matching torchvision install and available cached weights/network. Without pretrained weights, the backbone is random initialized; results must not be described as transfer learning.

## Training methodology

Integer class labels use cross-entropy loss. AdamW, weight decay, validation-based learning-rate reduction, and best validation checkpoint selection are implemented. Dataset loading streams batches from disk. The measured baseline used the custom CNN, CPU device, seed 48, one epoch, batch size 16, 64×64 input, and two CPU threads. Training metrics were loss 1.3160 / accuracy 0.5199; validation metrics were loss 1.3325 / accuracy 0.6173. This short run provides a runnable baseline only and is not comparable to a converged model.

## Evaluation methodology

The selected checkpoint is evaluated on the test-driver partition once. Reported measures include test loss and accuracy, macro precision/recall/F1, zero-division-safe per-class metrics, confusion matrix, support, split counts, parameter count, checkpoint bytes, and mean model forward latency. History curves, confusion matrix, per-class F1, JSON metrics, and text report are generated only by successful actual runs. Test results must not influence model selection.

## Results

| Result | Value |
|---|---|
| Dataset manifest / driver split | Complete: 22,424 images, 26 drivers, zero split overlap |
| Dataset split counts | Train: 15,340 images / 18 drivers; validation: 3,316 / 4; test: 3,768 / 4 |
| Training configuration | Custom CNN; 1 epoch; 64×64; batch 16; CPU, 2 threads |
| Training loss / accuracy | 1.3160 / 0.5199 |
| Validation loss / accuracy | 1.3325 / 0.6173 |
| Held-out test loss / accuracy | 0.8993 / 0.6789 on 3,768 images (4 unseen drivers) |
| Macro precision / recall / F1 | 0.7233 / 0.6720 / 0.6586 |
| Parameters / parameter storage | 532,914 / 2,131,656 bytes (float32 parameter storage) |
| Checkpoint file / device | 6,438,929 bytes / CPU |
| Mean forward latency | 6.38 ms/image (batched evaluation on CPU; input 64×64) |

### Per-class test results

| Code | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| c0 | 0.9320 | 0.2487 | 0.3926 | 386 |
| c1 | 0.9454 | 0.6410 | 0.7640 | 351 |
| c2 | 0.7397 | 0.8080 | 0.7723 | 401 |
| c3 | 0.9772 | 0.8706 | 0.9208 | 394 |
| c4 | 0.9117 | 0.7207 | 0.8050 | 401 |
| c5 | 0.7440 | 0.9853 | 0.8478 | 407 |
| c6 | 0.8267 | 0.7434 | 0.7828 | 417 |
| c7 | 0.4845 | 1.0000 | 0.6527 | 328 |
| c8 | 0.3455 | 0.1970 | 0.2510 | 335 |
| c9 | 0.3265 | 0.5057 | 0.3968 | 348 |

No numerical experimental results are fabricated.

## Discussion

Driver-level grouping offers a more defensible estimate for unseen drivers than random image-level splitting, which can place correlated images of the same individual in both fitting and evaluation data. This one-epoch test score is preliminary: the model under-detects c0 (recall 0.2487), c8 (0.1970), and c9 (0.5057), while predicting c7 with recall 1.0 but modest precision (0.4845). The confusion matrix and per-class report are saved as artifacts. The substantial gap among classes and short training schedule indicate more training/model selection is needed; do not rely on these predictions. The test set is held out and must not be reused for tuning. CPU-only training is slow. MobileNetV2 requires a compatible torchvision build; pretrained weights require access or a local cache.

## Limitations

The corpus is curated and does not validate performance under changing vehicles, camera positions, demographics, lighting, or real traffic. This baseline uses just one epoch and low 64×64 resolution, with a single driver-group split; results have variance and do not establish reliable deployment performance. Image classification does not provide temporal context. The UI is demonstrative only. Further training/model comparisons should use validation drivers for model selection and preserve the test results as a final holdout.

## Conclusion

The project provides a driver-disjoint PyTorch workflow and reports measured results for one short baseline. The initial run achieved 67.89% accuracy and 0.6586 macro F1 on held-out drivers, but class-level weaknesses and limited training make it unsuitable for operational use. Longer validation-selected experiments are needed; never treat it as a driving safety system.

## References

- State Farm. *State Farm Distracted Driver Detection*. Kaggle competition page. https://www.kaggle.com/c/state-farm-distracted-driver-detection (accessed 2026-10-23). Dataset descriptions and access conditions should be checked against the live competition page.
- Additional model and research references: **pending bibliographic verification before citation**.
