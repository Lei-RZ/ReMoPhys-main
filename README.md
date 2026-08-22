# ReMoPhys

**Dual-Spectrum Multiparameter Physiological Sensing at High Altitudes**

ReMoPhys is a dual-spectrum remote physiological sensing framework for
estimating **heart rate (HR)**, **blood oxygen saturation (SpO2)**, and
**respiration rate (RR)** from synchronized RGB and infrared facial
videos. The framework is designed for high-altitude monitoring, with
particular emphasis on physiological recovery after exercise.

This repository builds on the training and evaluation pipeline used by
**FusionVitals / FusionPhysNet** and **rPPG-Toolbox**.

> **Naming note:** the paper and dataset use the name **LADH+**. The
> implementation directory in this repository still keeps the legacy
> name `rppg_tool_LADH_SE`.

## 📖 Overview

Remote photoplethysmography (rPPG) enables non-contact physiological
measurement from facial videos. High-altitude environments introduce
additional challenges because cardiovascular, respiratory, and
oxygen-related signals vary with altitude exposure, motion, exercise,
and post-exercise recovery.

ReMoPhys processes synchronized RGB and infrared facial clips and
jointly estimates:

- **HR** from the reconstructed BVP/rPPG waveform;
- **SpO2** through scalar regression from the reconstructed BVP
  representation;
- **RR** from the reconstructed respiration waveform.

The framework combines temporal-difference representation learning,
adaptive dual-spectrum aggregation, task-specific experts, and
multi-scale temporal modeling.

## ✨ Main Contributions

- **LADH+ Dataset**: extends the original LADH collection with seven
  additional young-adult participants while preserving synchronized RGB,
  infrared, BVP, SpO2, and respiration acquisition.
- **High-altitude recovery evaluation**: the newly collected data
  introduce an intensified State 5 exercise procedure before recovery
  recording.
- **TemporalDiffConv3d + SE**: explicitly models local temporal
  variation while recalibrating channel responses.
- **Cross Spectrum Adaptive Aggregation (CSAA)**: predicts local
  normalized RGB/IR modality weights and refines the fused physiological
  representation.
- **TaskMoE adapters**: uses separate low-rank mixture-of-experts
  adapters for the BVP and respiration pathways.
- **Multi-Scale Periodic Temporal Mixer**: models periodic physiological
  patterns with temporal dilation rates `1`, `2`, `4`, and `8`.
- **Multiparameter prediction**: supports BVP/HR, SpO2, and
  respiration/RR estimation under RGB-only, IR-only, and RGB+IR
  settings.

## 🧠 ReMoPhys Architecture

The main implementation is located at:

``` text
rppg_tool_LADH_SE/neural_methods/model/ReMoPhysNet.py
```

The trainer selects the proposed model with:

``` yaml
MODEL:
  NAME: ReMoPhysNet
  MultiPhysNet:
    FRAME_NUM: 256
```

The overall data flow is:

``` text
RGB facial clip ──> RGB temporal encoder ──┐
                                           │
                                           ├─> Cross Spectrum Adaptive Aggregation
                                           │              │
IR facial clip  ──> IR temporal encoder  ──┘              │
                                                          │
                                  ┌───────────────────────┴───────────────────────┐
                                  │                                               │
                           BVP TaskMoE Adapter                              RR TaskMoE Adapter
                                  │                                               │
                           Periodic Signal Head                            Periodic Signal Head
                                  │                                               │
                           BVP/rPPG Waveform                               Respiration Waveform
                              │          │                                         │
                              HR       SpO2                                        RR
```

### Core Modules

**1. TemporalDiffConv3d**

A scaled temporal-difference residual is added to the input before 3D
convolution. The encoder combines this operation with BatchNorm, ReLU,
and squeeze-and-excitation blocks.

**2. Cross Spectrum Adaptive Aggregation**

The RGB and infrared feature maps are spatially and temporally aligned
when necessary. The current implementation concatenates the two encoded
representations, predicts two local modality-weight maps with a
convolutional gate, normalizes them with Softmax, performs weighted
aggregation, and applies residual feature refinement.

**3. TaskMoEAdapter**

The shared physiological representation is adapted separately for BVP
and respiration. Each adapter contains four low-rank experts and a
task-specific router that predicts normalized expert coefficients.

**4. MultiScalePeriodicTemporalMixer**

The BVP and respiration branches use parallel depthwise 1D temporal
convolutions with dilation values `1`, `2`, `4`, and `8`, followed by
feature mixing and residual addition.

**5. SpO2 Regression Head**

SpO2 is estimated from the reconstructed BVP/rPPG waveform. In the
current implementation, the output is bounded to the range `[85, 100]`.

## 🗂 Repository Structure

``` text
ReMoPhys-main/
├── README.md
└── rppg_tool_LADH_SE/
    ├── main.py
    ├── config.py
    ├── setup.sh
    ├── requirements.txt
    ├── configs/
    │   └── train_configs/
    │       ├── LADH+_PHYSNET_face_RGB_IR_both_d.yaml
    │       ├── LADH+_PHYSNET_face_RGB_IR_both_p.yaml
    │       ├── LADH+_PHYSNET_face_RGB_IR_bvp_*.yaml
    │       ├── LADH+_PHYSNET_face_RGB_IR_spo2_*.yaml
    │       └── LADH+_PHYSNET_face_RGB_IR_rr_*.yaml
    ├── dataset/
    │   └── data_loader/
    │       └── LADHLoader.py
    ├── neural_methods/
    │   ├── model/
    │   │   ├── ReMoPhysNet.py
    │   │   ├── MultiPhysNet.py
    │   │   └── FusionPhysNet.py
    │   └── trainer/
    │       └── MultiPhysNetTrainer.py
    └── evaluation/
```

## 🔬 LADH+ Dataset

LADH+ is a unified high-altitude physiological sensing dataset that
integrates the original LADH recordings with newly collected participant
data.

The complete collection contains **308 synchronized RGB and infrared
videos** and retains the five-state monitoring protocol used by LADH.

The newly collected portion adds **seven participants**:

- three participants recorded over ten consecutive days;
- four participants recorded on a single day for independent
  subject-based evaluation.

### Recording Setup

| Modality / Signal           | Device / Setting                           |
|-----------------------------|--------------------------------------------|
| RGB + infrared facial video | WN-12207K3321SM290 camera                  |
| Video resolution            | 640 × 480                                  |
| Video frame rate            | 30 FPS                                     |
| BVP / SpO2 reference        | CMS50E pulse oximeter                      |
| PPG sampling rate           | 20 Hz                                      |
| Respiration reference       | HKH-11C respiratory sensor                 |
| Respiration sampling rate   | 50 Hz                                      |
| Synchronization             | Millisecond-level synchronized acquisition |

An example of the synchronized acquisition setup inherited from LADH is
available in:

``` text
rppg_tool_LADH_SE/images/collection.png
```

### Five Recording States

1.  **State 1** — seated rest;
2.  **State 2** — seated personal care;
3.  **State 3** — standing rest;
4.  **State 4** — standing personal care;
5.  **State 5** — post-exercise recovery.

For the newly collected LADH+ recordings, State 5 uses a more intensive
exercise procedure involving repeated field running and rapid stair
ascent/descent before the recovery recording.

### Expected Data Organization

`LADHLoader.py` searches for participant folders named `p_*` and state
folders such as `v01`–`v05`. A typical layout is:

``` text
data/LADH+/
├── day_01/
│   ├── p_xxx/
│   │   ├── v01/
│   │   │   ├── BVP.csv
│   │   │   ├── RR.csv
│   │   │   ├── SpO2.csv
│   │   │   ├── frames_timestamp_RGB.csv
│   │   │   ├── frames_timestamp_IR.csv
│   │   │   ├── video_RGB_H264.avi
│   │   │   └── video_IR_H264.avi
│   │   ├── v02/
│   │   ├── v03/
│   │   ├── v04/
│   │   └── v05/
│   └── ...
├── day_02/
└── ...
```

The loader synchronizes BVP, SpO2, and respiration references to
video-frame timestamps before preprocessing.

## 🔧 Setup

Enter the implementation directory first:

``` bash
cd rppg_tool_LADH_SE
```

### Option A: Repository Setup Script

The repository inherits the FusionVitals/rPPG-Toolbox setup workflow:

``` bash
bash setup.sh
conda activate rppg-toolbox
pip install -r requirements.txt
```

> **Environment note:** the current `setup.sh` creates an environment
> with PyTorch 1.12.1 and CUDA Toolkit 10.2. The experiments reported in
> the ReMoPhys paper used **Python 3.8, PyTorch 2.2.2, and CUDA 12.1**
> on an **NVIDIA A100**. For paper-level reproduction, use the paper
> environment rather than relying on the inherited PyTorch version in
> `setup.sh`.

### Option B: Paper Environment

Create a Python 3.8 environment, install a PyTorch 2.2.2 build
compatible with CUDA 12.1, and then install the remaining dependencies:

``` bash
conda create -n remophys python=3.8 -y
conda activate remophys

# Install PyTorch 2.2.2 with CUDA 12.1 using the appropriate PyTorch package.
pip install -r requirements.txt
```

## ⚙️ Training ReMoPhys

The main dual-spectrum day-based configuration is:

``` text
./configs/train_configs/LADH+_PHYSNET_face_RGB_IR_both_d.yaml
```

Although the filenames retain `PHYSNET` for compatibility with the
original project, the proposed model is selected by:

``` yaml
MODEL:
  NAME: ReMoPhysNet
```

### Step 1: Update Dataset Paths

Edit the YAML file and replace the original machine-specific paths:

``` yaml
TRAIN:
  DATA:
    DATA_PATH: /path/to/LADH+/*
    CACHED_PATH: /path/to/cache/train

VALID:
  DATA:
    DATA_PATH: /path/to/LADH+/*
    CACHED_PATH: /path/to/cache/valid

TEST:
  DATA:
    DATA_PATH: /path/to/LADH+/*
    CACHED_PATH: /path/to/cache/test
```

### Step 2: Preprocess the Dataset

For the first run:

``` yaml
DO_PREPROCESS: True
```

After preprocessing has been completed and the cached data/file lists
have been generated, set:

``` yaml
DO_PREPROCESS: False
```

to avoid repeating preprocessing.

### Step 3: Train and Test

``` bash
python main.py \
  --config_file ./configs/train_configs/LADH+_PHYSNET_face_RGB_IR_both_d.yaml \
  --r_lr 9e-3 \
  --epochs 30 \
  --path ./results/LADH+_RGB_IR_both
```

The main paper settings are:

| Setting          |                  Value |
|------------------|-----------------------:|
| Learning rate    |                 `9e-3` |
| Epochs           |                   `30` |
| Batch size       |                   `16` |
| Clip length      |           `256` frames |
| Input resolution |              `72 × 72` |
| Video frame rate |               `30 FPS` |
| Optimizer        |                   Adam |
| Model selection  | lowest validation loss |

When `TEST.USE_LAST_EPOCH: False`, the checkpoint with the lowest
validation loss is used for testing.

## 🧾 YAML Configuration

The training and evaluation pipeline is controlled by YAML files.

### `TOOLBOX_MODE`

- `train_and_test`: train the model and evaluate the selected
  checkpoint;
- `only_test`: evaluate a checkpoint specified by
  `INFERENCE.MODEL_PATH`.

### `TASK`

- `bvp`: reconstruct BVP/rPPG and derive HR;
- `spo2`: estimate SpO2;
- `rr`: reconstruct respiration and derive RR;
- `both`: jointly optimize BVP, SpO2, and respiration.

### `DATASET_TYPE`

- `face`: RGB only;
- `face_IR`: infrared only;
- `both`: synchronized RGB + infrared.

### `TRAIN / VALID / TEST`

Important fields include:

- `DATA.INFO.STATE`: recording states, e.g. `[1, 2, 3, 4, 5]`;
- `DATA.INFO.TYPE`: `1` for RGB and `2` for infrared;
- `DATA.DATASET_TYPE`: `face`, `face_IR`, or `both`;
- `DATA_PATH`: raw-data path;
- `CACHED_PATH`: preprocessed-data path;
- `BEGIN` / `END`: split range used by the loader;
- `DO_PREPROCESS`: enable/disable preprocessing;
- `DATA_TYPE`: video preprocessing mode;
- `LABEL_TYPE`: physiological-label preprocessing mode;
- `DO_CHUNK`: split recordings into clips;
- `CHUNK_LENGTH`: number of frames per clip;
- `CROP_FACE`: face-cropping configuration;
- `RESIZE.H` / `RESIZE.W`: spatial input size.

### `METRICS`

The provided configs include:

``` yaml
TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA']
```

The paper mainly reports MAE and MAPE for HR, SpO2, and RR, with Pearson
correlation and RMSE additionally emphasized for the focused State 5
SpO2 evaluation.

## 🧪 Evaluation Protocols

The paper evaluates ReMoPhys under complementary protocols:

| Protocol                     | Training                                 | Validation                               | Test                                    |
|------------------------------|------------------------------------------|------------------------------------------|-----------------------------------------|
| Original LADH subject-based  | 8 participants                           | 3 different participants                 | 10 unseen participants                  |
| Original LADH day-based      | 7 complete days                          | 2 complete days                          | 1 unseen day                            |
| State 5 subject transfer     | 8 original-LADH participants, all states | 3 original-LADH participants, all states | State 5 from 4 newly added participants |
| Complete LADH+ subject-based | 8 original-LADH participants             | 3 different original-LADH participants   | 4 newly added one-day participants      |
| Complete LADH+ day-based     | 7 complete days                          | 2 complete days                          | 1 complete day                          |

For participant-based experiments, the provided `_p.yaml` files use
separately prepared train/validation/test paths. Check the `TASK` field
inside each YAML before running, because some filenames are retained
from earlier experiment templates.

## 📊 Main Results

### Complete LADH+

| Protocol | Model                  | HR MAE ↓ | HR MAPE ↓ | SpO2 MAE ↓ | SpO2 MAPE ↓ | RR MAE ↓ | RR MAPE ↓ |
|----------|------------------------|---------:|----------:|-----------:|------------:|---------:|----------:|
| Day      | FusionPhysNet baseline |     4.82 |      4.93 |       1.35 |        1.42 |     2.31 |     11.01 |
| Day      | **ReMoPhys**           | **4.07** |  **4.23** |   **1.30** |    **1.37** | **2.18** |     11.06 |
| Subject  | FusionPhysNet baseline |     6.09 |      7.51 |       1.92 |        2.09 |     2.53 |     10.81 |
| Subject  | **ReMoPhys**           | **4.70** |  **6.11** |   **1.51** |    **1.64** | **2.48** | **10.24** |

### Focused SpO2 Evaluation on State 5

| Protocol | Model                  |     MAE ↓ |    MAPE ↓ | Pearson ↑ |    RMSE ↓ |
|----------|------------------------|----------:|----------:|----------:|----------:|
| Subject  | FusionPhysNet baseline |     1.848 |     1.985 |     0.315 |     2.382 |
| Subject  | **ReMoPhys**           | **1.738** | **1.866** | **0.414** | **2.311** |
| Day      | FusionPhysNet baseline |     1.713 |     1.853 |     0.324 |     1.930 |
| Day      | **ReMoPhys**           | **1.572** | **1.699** | **0.507** | **1.730** |

These results show lower SpO2 estimation error and stronger correlation
with the reference measurements during high-altitude post-exercise
recovery.

## 🧩 Ablation Study

The paper evaluates the contribution of the four main architectural
components under the complete LADH+ subject protocol.

| Model           | TDConv | CSAA | TaskMoE | MSMixer | HR MAE ↓ | SpO2 MAE ↓ | RR MAE ↓ |
|-----------------|:------:|:----:|:-------:|:-------:|---------:|-----------:|---------:|
| Baseline        |   ✗    |  ✗   |    ✗    |    ✗    |     6.09 |       1.92 |     2.53 |
| Without TDConv  |   ✗    |  ✓   |    ✓    |    ✓    |     5.01 |       1.59 |     2.59 |
| Without CSAA    |   ✓    |  ✗   |    ✓    |    ✓    |     5.49 |       1.86 |     2.88 |
| Without TaskMoE |   ✓    |  ✓   |    ✗    |    ✓    |     5.37 |       1.60 |     2.65 |
| Without MSMixer |   ✓    |  ✓   |    ✓    |    ✗    |     4.97 |       1.85 |     2.64 |
| **ReMoPhys**    |   ✓    |  ✓   |    ✓    |    ✓    | **4.70** |   **1.51** | **2.48** |

The complete model achieves the lowest MAE for all three physiological
targets among the evaluated component configurations.

## ⚠️ Repository-Specific Notes

Before running the uploaded code on a new machine, please check the
following legacy items:

1.  **LADH+ dataset identifier**  
    The provided `LADH+_*.yaml` files use `DATASET: LADH+`, while
    `main.py` currently contains inconsistent legacy dataset-name
    handling in the training/validation branches. Make sure `LADH+` is
    mapped to `LADHLoader` consistently for `TRAIN`, `VALID`, and `TEST`
    before running these configs.

2.  **Machine-specific output paths**  
    `neural_methods/trainer/MultiPhysNetTrainer.py` contains several
    absolute `/gpfs/home/...` CSV output paths inherited from the
    development environment. Replace or remove these paths before
    running on another machine.

3.  **Setup script vs. paper environment**  
    `setup.sh` installs PyTorch 1.12.1 with CUDA Toolkit 10.2, whereas
    the paper experiments use PyTorch 2.2.2 with CUDA 12.1.

4.  **Legacy config filenames**  
    Some config filenames are inherited from earlier experiments. Always
    verify the actual `TASK`, `DATASET_TYPE`, dataset paths, and split
    settings inside the YAML file.

5.  **Pretrained checkpoints**  
    Pretrained ReMoPhys checkpoints are not included in the current
    repository archive.

## 🙏 Acknowledgements

This codebase is developed from the experimental pipeline used in
**FusionVitals / FusionPhysNet** and is based on **rPPG-Toolbox** for
remote physiological sensing training, preprocessing, and evaluation.

Please also cite the corresponding LADH/FusionVitals and rPPG-Toolbox
papers when using their dataset, code, or evaluation framework.

## 📝 Citation

If you use ReMoPhys in your research, please cite:

> **ReMoPhys: Dual-Spectrum Multiparameter Physiological Sensing at High
> Altitudes**

The final BibTeX entry can be added after the publication information is
available.
