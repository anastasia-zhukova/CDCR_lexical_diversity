# train on  small data
python mtl_train.py --data_path small_mixed_dataset --d_model 64 --tqdm_print_step 10 --training_strategy simultaneous --batch_size 8 --val_batch_size 8 --epochs 7

# eval on  small data
python mtl_eval.py --data_path small_mixed_dataset --tqdm_print_step 10 --model_name mtl_model_2023-08-07_dmodel_64_experiment_XX_epoch6_of_6 --batch_size 8

## Data process steps:

1. Run the data_preparation.py script

This code converts datasets into tensors using PyTorch. The script provides functions to load and process the data, including adding start and end tags, tokenization, and saving the resulting tensors. The user can interactively choose a dataset to convert and save the tensors in the `individual_datasets` folder.

The processed tensors for each dataset will be saved in the `individual_datasets` folder, with specific file names for each split.

