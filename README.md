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