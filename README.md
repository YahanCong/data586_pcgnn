# PC-GNN

This is data586 project about reproduce  "[Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection](https://dl.acm.org/doi/abs/10.1145/3442381.3449989)" (WebConf 2021).


## Requirements

```
argparse          1.1.0
networkx          1.11
numpy             1.16.4
scikit_learn      0.21rc2
scipy             1.2.1
torch             1.4.0
```

## Dataset

We have three dataset: YelpChi, FDCompCN(comp) and AlibabaLoan. Please put them in `/data` directory and run `unzip /data/YelpChi.zip`, `unzip /data/comp.zip`, and `unzip /data/AlibabaLoan.zip` to unzip the datasets.

Run `python src/data_process.py` to pre-process the data.


## Usage

YelpChi dataset (3-relation dataset):

```sh
python main.py --config ./config/pcgnn_yelpchi.yml
```

FDCompCN(comp) dataset (3-relation dataset):

```sh
python main.py --config ./config/pcgnn_comp.yml
```

AlibabaLoan dataset (4-relation dataset):
```sh
python main.py --config ./config/pcgnn_ali.yml
```
## Note for Alibaba dataset.

Since we have four relations in Alibaba, before you run `python main.py --config ./config/pcgnn_ali.yml`, please uncomment all codes with keyword `r4` (r4_score, r4_list, r4_sample_num_list, etc) in `src/layers`, and uncomment all code with keyword `relation4` in `src/model_handler` to replace code for 4-relation dataset. 
