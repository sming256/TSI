# TSI
This repo holds the official pytorch implementation of paper: ["TSI: Temporal Scale Invariant Network for Action Proposal Generation"](https://openaccess.thecvf.com/content/ACCV2020/papers/Liu_TSI_Temporal_Scale_Invariant_Network_for_Action_Proposal_Generation_ACCV_2020_paper.pdf), which is accepted in ACCV 2020.

- Author: Shuming Liu, Xu Zhao, Haisheng Su, Zhilan Hu

## Environment
This code is built on `pytorch1.10+CUDA11`, but other version may also be fine.

To install the dependency: `pip install numpy pandas easydict tqdm scipy h5py PyYAML`

## Data Preparation
Change the feature root and feature name in the config files, such as in `config/anet/anet_tsn.yaml`.

### For ActivityNet dataset
#### TSN feature
- Download the provided feature in ["BSN"](https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch#download-datasets). The data path should be `DATAPATH/features/tsn_anet/csv_mean_100/v_224E-VtB4k4.csv`
- `FEATURE.name` in config should be `tsn_anet`

#### TSP feature
- Download the provided feature in ["TSP"](https://github.com/HumamAlwassel/TSP/tree/master#pre-extracted-tsp-features). The data path should be `DATAPATH/features/tsp_r2plus1d_34/csv_unresize/v_224E-VtB4k4.csv`
- `FEATURE.name` in config should be `tsp_r2plus1d_34`

### For THUMOS dataset
#### TSN feature
- use the provided feature in ["G-TAD"](https://github.com/frostinassiky/gtad#data-setup). The data path should be `DATAPATH/features/tsn_gtad/rgb_train.h5`
- `FEATURE.name` in config should be `tsn_gtad`

#### I3D feature in P-GCN
- use the provided feature in ["P-GCN"](https://github.com/Alvin-Zeng/PGCN#download-features). The data path should be `DATAPATH/features/i3d_pgcn_snippet8/RGB/`
- `FEATURE.name` in config should be `i3d_pgcn_snippet8`

#### I3D feature in BU-TAL
- use the provided feature in ["BU-TAL"](https://github.com/PeisenZhao/Bottom-Up-TAL-with-MR#data-preparation). The data path should be `DATAPATH/features/i3d_butal_snippet4_clip16/video_validation_0000051.npy`
- `FEATURE.name` in config should be `i3d_butal_snippet4_clip16`

### For HACS dataset
#### SlowFast feature
- use the provided feature in ["TCANet"](https://github.com/qinzhi-0110/Temporal-Context-Aggregation-Network-Pytorch#download-datasets). The data path should be `DATAPATH/features/slowfast101/pkl_unresize/0_0MMzh2E3U.pkl`
- `FEATURE.name` in config should be `slowfast101/pkl_unresize`

## Train and Test
### Bash Run
```
bash train.sh {config_path} {GPU_num}
```

For example: `bash train.sh configs/anet/anet_tsn.yaml 1`

### Step-by-Step Run
```
python scripts/train.py {config_path} {GPU_num}
python scripts/test.py  {config_path} {GPU_num} {checkpoint_path}
python scripts/post.py  {config_path}
```
- `{GPU_num}` is the GPU number used for training and inference.
- `{config_path}` is the path of config.
- `{checkpoint_path}` is the path of loading checkpoint. If empty, load the best loss checkpoint by default.

## Performance
Pretrained weights and experiment outputs can be found in [Google Drive](https://drive.google.com/drive/folders/1Tmll52C3Jdq7_ccYgWLLNpN9KnODN7rc?usp=sharing).

### ActivityNet 1.3
#### TSN feature provided by BSN

| Method | AR@1  | AR@5  | AR@10 | AR@100 |  AUC  |
| :----: | :---: | :---: | :---: | :----: | :---: |
|  BMN   | 33.60 | 49.28 | 56.71 | 75.33  | 67.26 |
|  TSI   | 32.86 | 49.69 | 57.47 | 75.47  | 68.24 |

#### TSP feature provided by TSP

| Method | AR@1  | AR@5  | AR@10 | AR@100 |  AUC  |
| :----: | :---: | :---: | :---: | :----: | :---: |
|  BMN   | 34.85 | 51.38 | 58.47 | 76.07  | 68.47 |
|  TSI   | 34.30 | 52.17 | 59.29 | 76.73  | 69.42 |

### THUMOS14
#### TSN feature provided by G-TAD

| Method | AR@50 | AR@100 | AR@200 | AR@500 | AR@1000 |  AUC  |
| :----: | :---: | :----: | :----: | :----: | :-----: | :---: |
|  BMN   | 40.61 | 49.79  | 57.40  | 65.75  |  70.72  | 62.08 |
|  TSI   | 40.93 | 50.23  | 57.88  | 66.46  |  71.95  | 62.99 |

#### I3D feature provided by P-GCN

| Method | AR@50 | AR@100 | AR@200 | AR@500 | AR@1000 |  AUC  |
| :----: | :---: | :----: | :----: | :----: | :-----: | :---: |
|  BMN   | 33.76 | 42.70  | 50.85  | 59.83  |  65.36  | 56.18 |
|  TSI   | 38.17 | 46.26  | 53.98  | 62.93  |  67.81  | 59.31 |

#### I3D feature provided by BU-TAL

| Method | AR@50 | AR@100 | AR@200 | AR@500 | AR@1000 |  AUC  |
| :----: | :---: | :----: | :----: | :----: | :-----: | :---: |
|  BMN   | 40.93 | 49.99  | 56.92  | 64.66  |  68.93  | 61.20 |
|  TSI   | 41.51 | 50.49  | 57.86  | 65.71  |  70.08  | 62.21 |

### HACS
#### SlowFast feature provided by TCANet

| Method | AR@1  | AR@5  | AR@10 | AR@100 |  AUC  |
| :----: | :---: | :---: | :---: | :----: | :---: |
|  BMN   | 19.90 | 40.00 | 49.22 | 70.52  | 61.66 |
|  TSI   | 19.38 | 41.13 | 50.87 | 71.83  | 63.25 |

## Acknowledgment and Citation
We thank for the help of Tianwei Lin, Doingqi Wang.

If you find this work is useful in your research, please consider citing:
```
@inproceedings{liu2020tsi,
  title={TSI: Temporal Scale Invariant Network for Action Proposal Generation},
  author={Liu, Shuming and Zhao, Xu and Su, Haisheng and Hu, Zhilan},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2020}
}
```

## Contact
For any question, please contact `sming256@gmail.com`.
