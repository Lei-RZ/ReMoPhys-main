# ReMoPhysNet on LADH-SE

This repository contains the implementation of **ReMoPhysNet** for dual-spectrum physiological perception on the **LADH-SE** dataset.

ReMoPhysNet uses synchronized RGB and infrared facial videos to estimate:

- Heart Rate (HR) from the predicted BVP waveform
- Blood Oxygen Saturation (SpO2)
- Respiration Rate (RR) from the predicted respiration waveform

The model contains spectrum-specific encoders, Reliability-Aware Fusion, and task-specific Mixture-of-Experts adapters.

# Setup

STEP1: `bash setup.sh`

STEP2: `conda activate rppg-toolbox`

STEP3: `pip install -r requirements.txt`

The environment used in the paper is:

- Python 3.8
- PyTorch 2.2.2
- CUDA 12.1
- NVIDIA A100

# Example of ReMoPhysNet Training

Please use config files under:

`./configs/train_configs/LADH_SE_PHYSNET_*`

Although the config filenames contain `PHYSNET`, the proposed model is selected by:

```yaml
MODEL:
  NAME: ReMoPhysNet
```

## Train ReMoPhysNet on LADH-SE with RGB + IR

STEP1: Prepare the LADH-SE dataset.

STEP2: Modify:

`./configs/train_configs/LADH_SE_PHYSNET_face_RGB_IR_both_d.yaml`

Change the following paths to your own dataset paths:

```yaml
DATA_PATH:
CACHED_PATH:
```

STEP3: Run:

```bash
python main.py --config_file ./configs/train_configs/LADH_SE_PHYSNET_face_RGB_IR_both_d.yaml --r_lr 9e-3 --epochs 30 --path res_30_9e-3/LADH_SE_RGB_IR
```

The paper uses:

- Learning rate: `9e-3`
- Epochs: `30`
- Batch size: `16`
- Clip length: `256`
- Input size: `72 x 72`
- Frame rate: `30 FPS`

Note1: Preprocessing is required only once. After preprocessing, set `DO_PREPROCESS: False` in the yaml file.

Note2: The model with the lowest validation loss is used for testing when `USE_LAST_EPOCH: False`.

Note3: You can modify the learning rate, epochs, dataset path, cache path, and result path.

# Yaml File Setting

The toolbox uses yaml files to control training and evaluation.

Important parameters:

* #### TOOLBOX_MODE
  * `train_and_test`: train the model and test using the best checkpoint.
  * `only_test`: test using the checkpoint specified by `INFERENCE.MODEL_PATH`.

* #### TASK
  * `bvp`: BVP waveform -> HR
  * `spo2`: SpO2 estimation
  * `rr`: respiration waveform -> RR
  * `both`: jointly estimate BVP, SpO2 and respiration

* #### DATASET_TYPE
  * `face`: RGB only
  * `face_IR`: infrared only
  * `both`: RGB + infrared

* #### TRAIN / VALID / TEST
  * `DATA.INFO.STATE`: recording states, e.g. `[1, 2, 3, 4, 5]`
  * `DATA.INFO.TYPE`: `1` for RGB and `2` for IR
  * `DATA.DATASET_TYPE`: `face`, `face_IR` or `both`
  * `DATA_PATH`: raw dataset path
  * `CACHED_PATH`: preprocessed data path
  * `BEGIN` / `END`: dataset split range
  * `DO_PREPROCESS`: whether preprocessing is enabled
  * `CHUNK_LENGTH`: number of frames in each clip
  * `CROP_FACE`: whether to crop the facial region

* #### MODEL
  Use:

```yaml
MODEL:
  NAME: ReMoPhysNet
  MultiPhysNet:
    FRAME_NUM: 256
```

* #### METRICS
  Example:

```yaml
TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA']
```

# LADH-SE Dataset

LADH-SE is the summer extension of LADH.

It contains synchronized RGB and infrared facial videos with BVP, SpO2 and respiration reference signals.

The complete LADH-SE collection contains **308 synchronized videos**.

Seven summer participants are added:

- 3 participants recorded for 10 consecutive days
- 4 participants recorded for one day for independent subject-based testing

The five recording states are:

1. Seated rest
2. Seated personal care
3. Standing rest
4. Standing personal care
5. Post-exercise recovery

A typical recording folder is organized as:

```text
data/LADH_SE/
└── p_xxx/
    ├── v01/
    │   ├── BVP.csv
    │   ├── HR.csv
    │   ├── RR.csv
    │   ├── SpO2.csv
    │   ├── frames_timestamp_IR.csv
    │   ├── frames_timestamp_RGB.csv
    │   ├── video_RGB_H264.avi
    │   └── video_IR_H264.avi
    ├── v02/
    ├── v03/
    ├── v04/
    └── v05/
```

# ReMoPhysNet

The ReMoPhysNet implementation is located at:

```text
neural_methods/model/ReMoPhysNet.py
```

The main model pipeline is:

```text
RGB Video ──> RGB Encoder ──┐
                            ├─> Reliability-Aware Fusion
IR Video  ──> IR Encoder  ──┘
                                  |
                     ┌────────────┴────────────┐
                     |                         |
               BVP MoE Adapter           RR MoE Adapter
                     |                         |
                rPPG Branch                RR Branch
                     |                         |
              BVP Waveform             Respiration Waveform
                |      |
                HR    SpO2                    RR
```

# LADH-SE Results

Results reported in the paper using ReMoPhysNet:

| Protocol | HR MAE | SpO2 MAE | RR MAE |
|---|---:|---:|---:|
| Day-based | 4.07 | 1.30 | 2.18 |
| Subject-based | 4.70 | 1.51 | 2.48 |

For the subject-based LADH-SE experiment, ReMoPhysNet reduces all six reported MAE/MAPE errors compared with FusionPhysNet.

# Citation

If you use this code or model, please cite:

```text
ReMoPhys for Dual Spectrum Physiological Perception in Daily Care Monitoring
```

The final BibTeX entry can be added after publication information is available.
