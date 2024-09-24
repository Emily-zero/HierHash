# protoHash

A Pytorch implementation of paper "Document Hashing with Multi-Grained Prototype-Induced Hierarchical Generative Model"(EMNLP 2024 findings).

### Main Dependencies

- python 3.7.16
- transformers 4.30.2
- torch 1.13.1
- kmeans-pytorch==0.3
- scikit-learn==0.24.2
- scipy==1.7.3

### Data
 We use three datasets: nyt, dbpedia and agnews. For efficiency, we run and store  the BERT feature of them.
 Since the data for BERT feature is extremely large, we didn't upload them all in this project. Instead, we upload the original data.
 One can refer to ./Data/example.csv for the format of data, and run their own features for this project.


```shell
# An example.
# Run on the NYT Ddataset, 16-bit setting.
CUDA_VISIBLE_DEVICES=5 python main.py nyt16 --data_path data/nyt_c2f --dataset BERT --seed 86548 --batch_size 64 --lr 0.0001 --encode_length 16 --cuda --workers 0 --max_length 200 --hiddim 128 --c2f --tau 0.3 --n-class 5 --hashing_alpha 1.0 --gumbel_temperature 1 --consis_temperature 1 --dropout_rate 0.3 --pooler_type cls --proto_tau 0.5 --yz_tau 1.0 --pretrain_epoch 20 --cond_ent_weight 0.1 --VAE_weight 1 --prob_weight 0 --KL_weight 1  --multi_queue --code_weight 1 --train --fine_cluster_nums 100
```

Also, one can refer to the `run.sh` for detailed running commands to reproduce the results reported in our paper.

