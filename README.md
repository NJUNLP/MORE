# MORE
This repository maintains the code and data for "MORE: A Multimodal Object-Entity Relation Extraction Dataset with a Benchmark Evaluation". (He, et al. ACM MM 2023).

## Data
Data can be downloaded from here **(we will upload it soon)**.

Before releasing the dataset publicly, we spent a lot of time rechecking all the data, removing most of the sensitive data, correcting obvious labeling errors, and adding some new data. As a result, we updated all the experimental results in the paper on the new dataset, and the updated version of the paper can be found here **(we will upload it soon)**.

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
sh run.sh \
    
## Citation
If you used the datasets or code, please cite our paper.

@inproceedings{he2023more, \
  title={MORE: A Multimodal Object-Entity Relation Extraction Dataset with a Benchmark Evaluation}, \
  author={He, Liang and Wang, Hongke and Cao, Yongchang and Wu, Zhen and Zhang, Jianbing and Dai, Xinyu}, \
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia}, \
  pages={4564--4573}, \
  year={2023} \
}
