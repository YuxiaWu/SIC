
This is the code for "Semi-supervised New Slot Discovery with Incremental Clustering", accepted by EMNLP 2022.

# 1. Data Preparation
- Camrest676: https://github.com/WING-NUS/sequicity/tree/master/data/CamRest676
- MultiWOZ_2.1: WOZ-hotel WOZ-attr https://github.com/budzianowski/multiwoz/tree/master/data 
- Cambridge SLU:  https://aspace.repository.cam.ac.uk/handle/1810/248271
- ATIS: the raw data: https://www.kaggle.com/siddhadev/atis-dataset  rasa format: https://github.com/howl-anderson/ATIS_dataset

# 2.Extracting candidate values

We use two frame semantic parsers SEMAFOR (http://www.cs.cmu.edu/~ark/SEMAFOR/) and Open-sesame(https://github.com/swabhs/open-sesame). 

To obtain the candidate values, follow the steps in https://github.com/vojtsek/joint-induction


# 3. Data preprocess for open slot discovery

python DataRepeat.py

We repeat the data when if it contains more than one slots.

The preprocessed data is save in the folder *data/*


# Model training

We implement our method using a public TEXTOIR toolkit which contains standard and unified interfaces to ensure fair comparison on different baselines https://github.com/thuiar/TEXTOIR. We extend the original interfaces to the open slot discovery task on our own datasets. 

- ./backbone/bert.py:  The value representation model use bert as backbone
- ./dataloaders/slot_loader.py: The data laoder
- ./methods: our methods are in the ./methods/semi_supervised/ours/
  - manager.py: including the train and eval process
  - pretrain.py: the pretrain process
  - semi-kmeans.py: the refined semi-kmeans method

Run:

```
CUDA_VISIBLE_DEVICES=2 bash run_SIC.sh
```

# Citation

```
@inproceedings{wu2022semi,
  title={Semi-supervised new slot discovery with incremental clustering},
  author={Wu, Yuxia and Liao, Lizi and Qian, Xueming and Chua, Tat-Seng},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2022},
  pages={6207--6218},
  year={2022}
}
```
