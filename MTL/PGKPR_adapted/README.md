# train on  small data
python mtl_train.py --data_path small_mixed_dataset --d_model 64 --tqdm_print_step 10 --training_strategy simultaneous --batch_size 8 --val_batch_size 8 --epochs 5

# eval on  small data
python mtl_eval.py --data_path small_mixed_dataset --tqdm_print_step 10 --model_name epoch6_loss_0.44943172653516134_acc_0.455535390199637_acc_0.8366666666666667 --batch_size 8