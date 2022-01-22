# BMN
This branch holds the baseline method, which is ["BMN: Boundary-Matching Network for Temporal Action Proposal Generation"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_BMN_Boundary-Matching_Network_for_Temporal_Action_Proposal_Generation_ICCV_2019_paper.pdf).

## Instructions
The environment setup, data preparation, training script are the same as default branch.

## Performance
Pretrained weights and experiment outputs can be found in [Google Drive](https://drive.google.com/drive/folders/1Tmll52C3Jdq7_ccYgWLLNpN9KnODN7rc?usp=sharing).

### ActivityNet 1.3
#### TSN feature provided by BSN 

| Method | AR@1  | AR@5  | AR@10 | AR@100 |  AUC  |
| :----: | :---: | :---: | :---: | :----: | :---: |
|  BMN   | 33.60 | 49.28 | 56.71 | 75.33  | 67.26 |

#### TSP feature provided by TSP

| Method | AR@1  | AR@5  | AR@10 | AR@100 |  AUC  |
| :----: | :---: | :---: | :---: | :----: | :---: |
|  BMN   | 34.85 | 51.38 | 58.47 | 76.07  | 68.47 |

### THUMOS14
#### TSN feature provided by G-TAD

| Method | AR@50 | AR@100 | AR@200 | AR@500 | AR@1000 |  AUC  |
| :----: | :---: | :----: | :----: | :----: | :-----: | :---: |
|  BMN   | 40.61 | 49.79  | 57.40  | 65.75  |  70.72  | 62.08 |

#### I3D feature provided by P-GCN

| Method | AR@50 | AR@100 | AR@200 | AR@500 | AR@1000 |  AUC  |
| :----: | :---: | :----: | :----: | :----: | :-----: | :---: |
|  BMN   | 33.76 | 42.70  | 50.85  | 59.83  |  65.36  | 56.18 |

#### I3D feature provided by BU-TAL

| Method | AR@50 | AR@100 | AR@200 | AR@500 | AR@1000 |  AUC  |
| :----: | :---: | :----: | :----: | :----: | :-----: | :---: |
|  BMN   | 40.93 | 49.99  | 56.92  | 64.66  |  68.93  | 61.20 |

### HACS
#### SlowFast feature provided by TCANet

| Method | AR@1  | AR@5  | AR@10 | AR@100 |  AUC  |
| :----: | :---: | :---: | :---: | :----: | :---: |
|  BMN   | 19.90 | 40.00 | 49.22 | 70.52  | 61.66 |
