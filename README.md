# MORE

This repository maintains the code and data for "MORE: A Multimodal Object-Entity Relation Extraction Dataset with a Benchmark Evaluation". (He, et al. ACM MM 2023).

## Data

Data can be downloaded from **[here](https://pan.baidu.com/s/1PA6raw1rbQKhPEL1FEEuAw?pwd=taft)**.

Before releasing the dataset publicly, we spent a lot of time rechecking all the data, removing most of the sensitive data, correcting obvious labeling errors, and adding some new data. As a result, we updated all the experimental results in the paper on the new dataset, and the updated version of the paper can be found here **(we will upload it soon)**.

In the revised version, MORE consists of 21 distinct relation types and contains 20,264 multimodal relational facts annotated on 3,559 pairs of textual titles and corresponding images. The dataset includes 13,520 visual objects (3.8 objects per image on average). We split the dataset into training, development, and testing sets consisting of 15,486, 1,742 and 3,036 facts respectively.

The overall result:

|  Model  | Accuracy  |  precision  |  Recall | F1-Ccore |
|  ----  | ----  | ---- | ---- | ---- |
| BERT+SG  | 61.79 | 29.61 | 41.27 | 34.48 |
| BERT+SG+Att | 63.74 | 31.10 | 39.28 | 34.71 |
| MEGA | 65.97 | 33.30 | 38.53 | 35.72 |
| IFAformer | 79.28 | 55.13 | 54.24 | 54.68 |
| MKGformer | 80.17 | 55.76 | 53.74 | 54.73 |
| VisualBERT | 82.84 | 58.18 | 61.22 | 59.66 |
| ViLBERT | **83.50** | **62.53** | 59.73 | 61.10 |
| **MOREformer** | **83.50** | 62.18 | **63.34** | **62.75** |



The performance on multiple entities/objects:

|    | Ent = 1, Obj = 1  |  Ent = 1, Obj > 1  |  Ent > 1, Obj = 1 | Ent > 1, Obj > 1 |
|  ----  | ----  | ---- | ---- | ---- |
| none Ratio  | 5% | 74.10% | 33.11% | 80.90% |
| Accuracy  | 80.00 | 84.57 | 67.55 | 84.43 |
| Precision  | 82.98 | 63.18 | 66.36 | 53.63 |
| Recall  | 82.11 | 66.31 | 72.28 | 52.47 |
| F1-Score  | 82.54 | 64.71 | 69.19 | 53.04 |



Macro-F1 on MORE dataset:

| Model Name  | Precision | Recall | Macro F1-Score |
|  ----  | ---- | ---- | ---- |
| MKGformer  | 48.50 | 42.60 | 43.57 |
| ViLBERT  | 49.13 | 49.37 | 48.66 |
| **MOREformer** | 51.02 | 50.66 | 50.02 |



Multi-object disambiguation analysis:

| Model Name  | Precision | Recall | F1-Score |
|  ----  | ---- | ---- | ---- |
| MKGformer  | 70.12/66.08 | 67.58/62.29 | 68.83/64.13 |
| ViLBERT  | **75.31/72.30** | 75.31/70.41 | 75.31/71.34 |
| **MOREformer** | 72.28/71.43 | **79.75/76.61** | **75.84/73.93** |


## Requirements

* transformers==4.11.3
* tensorboardX==2.4
* pytorch-crf==0.7.2
* torchvision==0.8.0
* scikit-learn==1.0
* numpy>=1.21
* tokenizers==0.10.3
* torch==1.7.0
* tqdm==4.49.0
* opencv-python==4.7.0.68
* ftfy==6.1.1

## Usage

conda create -n moreformer python==3.7 \
conda activate moreformer \
pip install -r requirements.txt \
sh run.sh
    
## Citation

If you used the datasets or code, please cite our paper.

```bibtex
@inproceedings{he2023more,
  title={MORE: A Multimodal Object-Entity Relation Extraction Dataset with a Benchmark Evaluation},
  author={He, Liang and Wang, Hongke and Cao, Yongchang and Wu, Zhen and Zhang, Jianbing and Dai, Xinyu},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4564--4573},
  year={2023}
}
```
