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

We have three dataset: YelpChi, FDComp and AlibabaLoan. Please put them in `/data` directory and run `unzip /data/YelpChi.zip`, `unzip /data/comp.zip`, and `unzip /data/AlibabaLoan.zip` to unzip the datasets.

Run `python src/data_process.py` to pre-process the data.


## Usage

```sh
python main.py --config ./config/pcgnn_yelpchi.yml
```

## Citation

```
@inproceedings{liu2021pick,
  title={Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection},
  author={Liu, Yang and Ao, Xiang and Qin, Zidi and Chi, Jianfeng and Feng, Jinghua and Yang, Hao and He, Qing},
  booktitle={Proceedings of the Web Conference 2021},
  pages={3168--3177},
  year={2021}
}
```

## Other implementations

Thanks for [Jack Huang](https://github.com/f26401004/PC-GNN) and [Ronald D. R. Pereira](https://github.com/ronaldpereira/pick-and-choose-gnn) for their kind implementations.
