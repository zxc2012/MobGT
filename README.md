## Installation steps

If you're using python virtual environment, make sure to create a venv folder with python `3.7` or `3.8`

```bash
cd MobGT/
python -m venv venv
source venv/bin/activate
```

Afterwards, run the following commands to install python packages:

```bash
pip install -U pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --force-reinstall
pip install torch_geometric
pip install wheel
pip install -r requirements.txt
```

## Data preparation

Please put and decompress the following dataset files into `dataset` directory at the <b><u>root</u></b> of the project.

- `foursquaregraph.7z`: Foursquare dataset
- `gowalla_nevda.7z`: Gowalla dataset
- `toyotagraph.7z`: Toyota dataset
- `poi-data.7z`: POI distance data files
- Copy all files from `dataset/baseline_models_dataset/` (`baseline_models_dataset.7z`)

## Project Organization

This project has been organized using 3 main folders.

1. `graphormer` - folder for the MobGT (STGTrans) project
2. `baseline_models` - baseline models for comparisson
3. `dataset` - all data used for training and testing both the proposed and baseline models

> NOTE: The commands for training and evaluation of each model are shown below. In order to illustrate how the output is, an example for one single epoch (epoch=1) is shown. However, the training files has as default greater number of epochs. Thus the results for more epochs will not be the same as the ones shown below.

### Training: Gowalla dataset

Warning: It takes around 20-30 minutes per epoch with Toyota dataset, so please try it with fewer epochs at first.

```bash
cd graphormer
python entry.py --num_workers 8 --seed 1 --batch_size 16 --dataset_name gowalla_nevda --gpus 1 --accelerator ddp --precision 16 --max_epochs 200 --ffn_dim 1024 --hidden_dim 128 --num_heads 8 --dropout_rate 0.1 --n_layers 6 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20 --check_val_every_n_epoch 4 --warmup_updates 40000 --tot_updates 400000 --default_root_dir exps/gowalla_nevda/
```

### Evaluation

Please just add `--test` argument to the training commands.

#### Example of output (Toyota, 1 epoch)

```
ACC @1: 0.0003, @5: 0.0009, @10: 0.0016
NDCG @1: 0.0003, @5: 0.0006, @10: 0.0008
MRR: 0.0013
```

If you want to train new model and disconsider the existent ones execute the command below:

```bash
rm -rf graphormer/exps/*
```

## Baseline Models

All the following baseline models have the same command to be train and tested. However, if it is desired to change the default parameters it, for now, necessary to change manually at each training file.

Dataset options `<dataset_type>`:

1. `toyota`
2. `gowalla`
3. `foursquare`

Regarding the evalution of baseline models, once the `train.py` be executed, it will show after the training the evaluation results (Acc@1, Acc@5, Acc@10, NDCG@1, NDCG@5, NDCG@10, MRR)

### Training and evaluation of LSTM

```bash
cd baseline_models/LSTM/
python train.py -dt <dataset_type ex: toyota>
```

#### Example of results output (Toyota)

```txt
==>Train Epoch:19 Loss:5.5005 lr:5e-05
==>Test [Acc@1, Acc@5, Acc@10, NCDG@1, NDCG@5, NDCG@10]: [0.13173492729382807, 0.24290396463973332, 0.2994192146315673, 0.13173492729382807, 0.18984936867627425, 0.20812322739393185] Loss::6.2744
==>Test [MRR]: 0.1893234137784137 Loss::6.2744
```

### Training and evaluation of LSTPM

```bash
cd baseline_models/LSTPM/
python train.py -dt <dataset_type ex: toyota>
```

#### Example of results output (Toyota)

```txt
epoch0: loss: tensor(5.8366, device='cuda:0', grad_fn=<DivBackward0>)
Scores:  [0.1329890142588846, 0.23854918428969, 0.2835428377197697, 0.1329890142588846, 0.18885384284651258, 0.20338215737195683, 0.1873599299935497]
```

### Training and evaluation of DeepMove

```bash
cd baseline_models/DeepMove/
python train.py -dt <dataset_type ex: toyota>
```

#### Example of results output (Toyota)

```txt
==>Test [Acc@1, Acc@5, Acc@10, NCDG@1, NDCG@5, NDCG@10]: [0.1236661545469111, 0.2456297602451762, 0.31114991424405164, 0.1236661545469111, 0.18675531110662474, 0.20797428140121066] Loss::6.0123
==>Test [MRR]: 0.18653992740492212 Loss::6.0123
single epoch time cost:17249.351979017258
```

### Training and evaluation of GetNext

```bash
cd baseline_models/GetNext/
python train.py -dt <dataset_type ex: toyota>
```

#### Example of results output (Toyota)

```
2023-09-13 18:49:57 Epoch 0/1
train_loss:16.8252, train_poi_loss:14.1238, train_time_loss:0.0664, train_cat_loss:2.0378, train_top1_acc:0.0275, train_top5_acc:0.0540, train_top10_acc:0.0681, train_top20_acc:0.0864, train_mAP20:0.0401, train_mrr:0.0431
val_loss: 21.6343, val_poi_loss: 18.7769, val_time_loss: 0.0799, val_cat_loss: 2.0589, val_top1_acc:0.0004, val_top5_acc:0.0013, val_top10_acc:0.0039, val_top20_acc:0.0079, val_mAP20:0.0013, val_mrr:0.0034,test_loss: 21.6343, test_poi_loss: 18.7769, test_time_loss: 0.0799, test_cat_loss: 2.0589, test_top1_acc:0.0004, test_top5_acc:0.0013, test_top10_acc:0.0039, test_top20_acc:0.0079, test_mAP20:0.0013, test_mrr:0.0034
```