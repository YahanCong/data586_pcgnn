# PC-GNN

This is data586 project about reproduce PC-GNN (WWW 2021).


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

We have three preprocessed datasets: [YelpChi](https://github.com/YahanCong/data586_pcgnn/blob/main/data/YelpChi.zip), [FDCompCN(comp)](https://github.com/YahanCong/data586_pcgnn/blob/main/data/comp.zip) and [AlibabaLoan](https://github.com/YahanCong/data586_pcgnn/blob/main/data/AlibabaLoan.zip). Please find them in `/data` directory and run `unzip /data/YelpChi.zip`, `unzip /data/comp.zip`, and `unzip /data/AlibabaLoan.zip` to unzip the datasets.

Run `python src/data_process.py` to pre-process the data.

The original dataset for yelpchi can be found in [YelpChi](https://odds.cs.stonybrook.edu/yelpchi-dataset/)

The original dataset for FDCompCN can be found in [FDCompCN](https://github.com/Split-GNN/SplitGNN/blob/master/data/FDCompCN.zip)

The original dataset about AliababaLoan can be found in [AliTianchi](https://tianchi.aliyun.com/dataset/168012) Our graph structure (sparse matrix) constructing process can be found in [Alibaba_data_transfer](https://github.com/YahanCong/data586_pcgnn/blob/main/data/Alibaba_data_transfer/fraud_data_transfer.ipynb)



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

Since we have four relations in Alibaba, before you run `python main.py --config ./config/pcgnn_ali.yml`, please uncomment all codes includes keyword `r4` (r4_score, r4_list, r4_sample_num_list, etc) in `src/layers`, and uncomment all code with keyword `relation4` in `src/model_handler` to switch the model for a 4-relation dataset.

## Parameter tunning

We use grid search to fine-tuning our models. Grid search results stores at `/grid`
