import argparse
from architecture import PGKPR
import torch
from data_loader import Phase, Batch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import os
import pandas as pd
import numpy as np
#Vocabulary utils
vocab_path = 'mscoco.vocab'

PAD_TAG = '[pad]'  # 0
UNK_TAG = '[unk]'
UN_KEY_TAG = '[unkey]'
START_TAG = '<s>'
END_TAG = '</s>'
SEP_TAG = '<sep>'

POS_TAG = ['[VB]',
           '[NNS]',
           '[RBR]',
           '[DT]',
           '[IN]',
           '[WRB]',
           '[VBZ]',
           '[NNP]',
           '[CD]',
           '[(]',
           '[NN]',
           '[VBG]',
           '[``]',
           '[PRP$]',
           "['']",
           '[VBD]',
           '[PDT]',
           '[CC]',
           '[POS]',
           '[LS]',
           '[$]',
           '[JJ]',
           '[.]',
           '[,]',
           '[#]',
           '[RB]',
           '[RP]',
           '[:]',
           '[VBP]',
           '[MD]',
           '[PRP]',
           '[)]',
           '[VBN]',
           '[FW]',
           '[SYM]']



def loadVocab(vocab_fpath, padTag=PAD_TAG, oov_mark=False, lowercase=True):
    vocab = [PAD_TAG, UNK_TAG, START_TAG, END_TAG, SEP_TAG, UN_KEY_TAG] + [i.lower() for i in POS_TAG]
    if not lowercase:
        vocab = [PAD_TAG, UNK_TAG, START_TAG, END_TAG, SEP_TAG, UN_KEY_TAG] + [i for i in POS_TAG]
    # if oov_mark:
    #     vocab += OOV_MAK
    vocab += [line.split()[0] for line in open(vocab_fpath, 'r', encoding="utf8").read().splitlines()]

    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    assert idx2token[0] == PAD_TAG
    return token2idx, idx2token


#################
### Train utils
#################

# Load tokenizer
LOWERCASE = True
token2idx, _ = loadVocab(vocab_path, lowercase=LOWERCASE)
vocab_size = len(token2idx)

def instantiate_model(args):
    model = PGKPR(d_model=args.d_model, vocab=vocab_size, mid_loss=args.mid_loss, double_pe=args.double_position,
                shared_emb=True)
    return model

    
    
def initialize_loss_and_optimizer(model):
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    return loss_function, optimizer

def parse_arguments_train():
    parser = argparse.ArgumentParser(description='Train PGKPR model.')
    parser.add_argument('--data_path', type=str, default="mixed_dataset/", help='Data path')
    parser.add_argument('--d_model', type=int, default=512, help='The model dimension.')
    parser.add_argument('--mid_loss', action='store_true', default=True, help='Whether to use mid loss.')
    parser.add_argument('--double_position', action='store_true', default=False, help='Whether to use double positional encoding.')
    parser.add_argument('--tqdm_print_step', type=int, default=1, help='The print step for tqdm.')
    parser.add_argument('--model_save_path', type=str, default='models/', help='The path to save the model checkpoints.')
    parser.add_argument('--training_strategy', type=str, default='sequential', 
                    choices=['sequential', 'sequential_randomized', 'simultaneous'], 
                    help='strategy: sequential, sequential_randomized, simultaneous')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=64, help='The batch size for validation.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--use_cpu', action='store_true', help='Whether to use CPU for training (overrides GPU).')
    return parser.parse_args()

def parse_arguments_eval():
    parser = argparse.ArgumentParser(description='Test PGKPR model.')
    parser.add_argument('--data_path', type=str, default="mixed_dataset/", help='Data path')
    parser.add_argument('--d_model', type=int, default=512, help='The model dimension.')
    parser.add_argument('--mid_loss', action='store_true', default=True, help='Whether to use mid loss.')
    parser.add_argument('--double_position', action='store_true', default=False, help='Whether to use double positional encoding.')
    parser.add_argument('--tqdm_print_step', type=int, default=1, help='The print step for tqdm.')
    parser.add_argument('--model_save_path', type=str, default='models/', help='The path to save the model checkpoints.')
    parser.add_argument('--model_name', type=str, help="The model's. name without .pkl")
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size for training.')
    parser.add_argument('--use_cpu', action='store_true', help='Whether to use CPU for training (overrides GPU).')
    return parser.parse_args()

def set_device(args):
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("use_cpu set to False. Running on", device)
    return device


def run_forward_pass(model, batch):
    predicted_probs = model.forward(src=batch.src, src_mask=batch.src_mask, dataset_id=batch.dataset_id)
    predicted_probs = predicted_probs.squeeze()
    return predicted_probs

def run_optimization_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def run_once(model, batch, loss_function, optimizer, is_training):
    predicted_probs = run_forward_pass(model, batch)
    loss=None
    if loss_function is not None:
        loss = loss_function(predicted_probs, batch.trg)
    
    if is_training and optimizer is not None:
        run_optimization_step(optimizer, loss)

    return loss,predicted_probs



def run_epoch(model, device, print_step, data_loader, loss_function, optimizer=None, is_training=True):
    total_loss = 0.0

    # Set the model in training or evaluation mode based on the 'is_training' flag
    model.train() if is_training else model.eval()

    with tqdm(total=len(data_loader)) as pbar:

        #Calculate accuracy here
        y_true = []
        y_pred = []

        for i, data in enumerate(data_loader):
            # Creates batch
            src, tgt, dataset_id = data
            src, tgt, dataset_id = src.to(device), tgt.to(device), dataset_id.to(device)
            batch = Batch(src, tgt, dataset_id)

            if is_training:
                binary_loss, predicted_probs = run_once(model, batch, loss_function, optimizer, is_training)
            else:
                # Use torch.no_grad() for validation to disable gradient computation
                with torch.no_grad():
                    binary_loss, predicted_probs = run_once(model, batch, loss_function, None, is_training)

            batch_prediction = predicted_probs.detach().cpu().numpy()

            if binary_loss: # Equal to None when evaluating on test data
                total_loss += binary_loss.item()
            y_true.extend(tgt.cpu().numpy())
            y_pred.extend(batch_prediction)

            if i % print_step == 0:
                pbar.update(print_step)
                if binary_loss:
                    pbar.set_description(f'Training loss: {binary_loss:.4f}' if is_training else f'Validation loss: {binary_loss:.4f}')

    epoch_loss = total_loss / len(data_loader)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred> 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_binary, average='macro')
    acc = accuracy_score(y_true, y_pred_binary)

    print(f"{'Training' if is_training else 'Validation'} loss: {epoch_loss:.4f}")
    print(f"{'Training' if is_training else 'Validation'} accuracy: {f1:.4f}")
    print(f"{'Training' if is_training else 'Validation'} f1: {acc:.4f}")

    return epoch_loss, f1, acc, y_true, y_pred


def save_predictions_to_csv(y_true, y_pred, output_folder,model_name):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Convert y_true and y_pred to DataFrames
    y_true_df = pd.DataFrame({'True Labels': y_true})
    y_pred_df = pd.DataFrame({'Predicted Labels': y_pred})

    # Concatenate both DataFrames along the columns
    predictions_df = pd.concat([y_true_df, y_pred_df], axis=1)

    # Save the predictions DataFrame to a CSV file
    output_file = os.path.join(output_folder, f'predictions_{model_name}.csv')
    predictions_df.to_csv(output_file, index=False)