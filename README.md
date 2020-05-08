# Classifying Gene Silencing From Cell Images

# Models
The following trained models can be downloaded from here: 
https://drive.google.com/drive/folders/1zci_EGxZVLnob3uF6CNT51Io0Mlu_5i-?usp=sharing

Included Models:
- ERM Densenet
- ERM Densenet Normalized
- ERM KaggleModel
- ERM KaggleModel Normalized
- IRM KaggleModel
- IRM KaggleModel Normalized
- Multitask
- Multitask Normalized

# Download Data
Run the following script (note the dataset is 60GB).
```
./download_data.sh
```
# Requriments 
We used AWS's `pytorch_p36` conda environment, with one tensorboard modification to make it run (`pip tensorboard ==1.14.0`)

We exported the requriments to `requirments.txt`, which could be installed with `pip install -r requriments.txt`

# Run Experiments
## Training
Run the following command:
```
python -W ignore train.py data/train/rxrx1/ kaggle erm test --num_epochs=100 --loss_scaling_factor=1 --normalization=experiment
```
arguments:  
data_dir: data/train/rxrx1/  
model_type: lr/cnn/densenet/kaggle/multitask  
train_type: erm/irm/multitask  
checkpoint_name: any name of file to save checkpoints  
--num_epochs: The number of epochs to train, default 100  
--loss_scaling_factor: The factor the loss is multiplied by before being added to the IRM penalty. A larger factor emphasizes classification accuracy over consistency across environments, default 1  
--normalization: Define normalization across a plate, experiment, or as none. csv with normalization values must be added to the data folder  

## Evaluation
```
python eval.py data/train/rxrx1/ kaggle saved_models/final_models/kaggle/irm_kaggle_no_norm_subset_finished.pth train  --sirna_selection=subset --eval_set=combined
```
data_dir: path to data. use data/train/rxrx1/  
model_type: use one of the following lr/cnn/densenet/kaggle/multitask  
model_path: path to saved, trained model  
dataset: "test" or "train"  
--sirna_selection: defines the selection of sirnas. It's either "subset", "full", or "control". Default = "subset"  
--eval_set: whether or not you want to evaluate holdout, used "combined" or "holdout". Default = "holdout"