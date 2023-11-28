## Unleash the Capabilities of the Vision-language Pre-training Model in Gaze Object Prediction
This repository is the official implementation of our paper
## Acknowledgement
Thanks Wang team for the wonderful open source project!https://github.com/CodeMonsterPHD/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction
## Data Preparation
note:
We use GOO-real to obtain object detection results and GOO-synth to obtain gaze object prediction results

The GOO dataset contains two subsets: GOO-Sync and GOO-Real. 

You can download GOO-synth from OneDrive:

Train:
[part1](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/ESk3KTXgDSFMmVcBW_EMAssBvxBf-eBSMK39g8yqFlRVgA?e=dSIXsd),
[part2](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EVoQzdFRQ85Bok0tmVpPLLcBk-adZBEcuRa-xp20sxVICw?e=hq5bwi),
[part3](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EUHIrsNLq_pHm2s7E8V9O5IBWBmvf5qI9jFue7wYOVNhyw?e=GYbTaW),
[part4](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EVTOQM6kkFJJm5v_h7K8r88BF2RlXa76DfE88v3rJkw70Q?e=NPuBzO),
[part5](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/ESLLYmaimxlIgCW4Hy_Q9LMBrMS6ouPsAwWDYf7kUStElA?e=bKvZYd),
[part6](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/ETevP-Nko6BAjBTidy-YsIYB8_145u7SeISKYtIr4AqKVg?e=0o9GOq),
[part7](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EUhglUujYbFIqYEOSgp5R5kBVZqvz0a4GdGogqlXXm1JAA?e=30ysd6),
[part8](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EY5AI0TLdrtMkN_pqV3FOzABtSPlnDOcjNT7QMEtDC5iXg?e=zI13Cp),
[part9](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EXHwQT0GRStFqSML6s8uW6kBLuFF8VjTf_opBg_ivtbUBQ?e=e0Agrv),
[part10](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EZ1LC_GF9a9EtUnofjLUtW4Bhb_-dg2bSgx6UYGS1piLrA?e=yawdDf),
[part11](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/ERLtx5hBM5tLtUPAjs2yvYgBQ8CrEW4S8zd9HKzow7jX2w?e=BTCvl6)



Test:
[GOOsynth-test_data](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EQ5DC43-aYVOt3-Jmy5ZIscBFd0yGvVeyiWOoHLFwZsJ4Q)

Annotation file:

[GOOsynth-train_data_Annotation](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/ET6h-STtBiVIn7jpwOP8bzEB__u-FEmwT8wXLKtTUw865g)

[GOOsynth-test_data_Annotation](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EW3HgRNFGPVHmaoYeFwhY1gB8UxwdEWSr55UwDaxwYvp1w)

You can download GOO-Real from OneDrive:

Train:
[GOOreal-train_data](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/ETjRM4To-QxNtp9a4hMTTVwBAfBJ8e6Hs7TNv5RzsJgj2w)

Test:
[GOOreal-test_data](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EXlBBBnd2khPk7S0_FKKGtQBCoIKoHSu7SmpdpxTAICpdw)

Annotation file:

[GOOreal-train_data_Annotation](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EZEpAPUtWSlGoS7DOwuV1G8BvhYuWZL_NLwqkDJe3sZ7JQ)

[GOOreal-test_data_Annotation](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/yangle_nwpu_mail_nwpu_edu_cn/EfCtqSoZxnxEgPKwB_IfmfcBvap1R8d3o8wjxUY6tPr0fw)

~~~~
Please ensure the data structure is as below
├── data
   └── goo_dataset
      └── goosynth
         └── picklefile
            ├── goosynth_test_v2_no_segm.pkl
            ├── goosynth_train_v2_no_segm.pkl
         └── test
            └── images
               ├── 0.png
               ├── 1.png
               ├── ...
               ├── 19199.png
         └── train
            └── images
               ├── 0.png
               ├── 1.png
               ├── ...
               ├── 172799.png
      └── gooreal
         └── train_data
            ├── 0.png
            ├── 1.png
            ├── ...
            ├── 3608.png
         └── test_data
            ├── 3609.png
            ├── 3610.png
            ├── ...
            ├── 4512.png
         ├── train.pickle
         ├── test.pickle

~~~~

## Environment Preparation
Requirements

```
conda env create -n qq -f environment.yaml
```

## Training & Inference

To carry out experiments on the GOO dataset, please follow these commands:

Experiments on GOO-Synth:
```train
python main.py --train_mode 0 
```
Experiments on GOO-Real:
```train
python main.py --train_mode 1 
```


