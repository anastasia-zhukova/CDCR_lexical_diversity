import argparse

import torch
from architecture import PGKPR
from data_loader import *

from tqdm import tqdm
from utils import loadVocab, vocab_path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train PGKPR model.')
    parser.add_argument('--d_model', type=int, default=512, help='The model dimension.')
    parser.add_argument('--mid_loss', action='store_true', default=True, help='Whether to use mid loss.')
    parser.add_argument('--double_position', action='store_true', default=False, help='Whether to use double positional encoding.')
    parser.add_argument('--tqdm_print_step', type=int, default=1, help='The print step for tqdm.')
    parser.add_argument('--model_save_path', type=str, default='models/', help='The path to save the model checkpoints.')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=64, help='The batch size for validation.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--use_cpu', action='store_true', help='Whether to use CPU for training (overrides GPU).')
    return parser.parse_args()

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
    loss = loss_function(predicted_probs, batch.trg)
    
    if is_training and optimizer is not None:
        run_optimization_step(optimizer, loss)

    return loss

def train_model():
    args = parse_arguments()

    # Load data
    data_path = "mixed_dataset/"
    train_dataloader = create_data_loaders(data_path=data_path, batch_size=args.batch_size, phase=Phase.TRAIN)
    val_dataloader = create_data_loaders(data_path=data_path, batch_size=args.val_batch_size, phase=Phase.VALIDATION)

    # Load tokenizer
    LOWERCASE = True
    token2idx, _ = loadVocab(vocab_path, lowercase=LOWERCASE)
    vocab_size = len(token2idx)

    # Search device
     # Set the device based on the 'use_cpu' argument
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("use_cpu set to False. Running on", device)

    # Load model and plug optimizer
    model = PGKPR(d_model=args.d_model, vocab=vocab_size, mid_loss=args.mid_loss, double_pe=args.double_position,
                  shared_emb=True)
    model.to(device)

    model_opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    dataloader = {'train': train_dataloader, 'val': val_dataloader}

    # Run training
    loss_function = torch.nn.BCELoss()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f'\nEPOCH {epoch}')
        model.train()
        total_loss = 0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                is_training = True
            else:
                model.eval()
                is_training = False

            with tqdm(total=len(dataloader[phase])) as pbar:
                for i, data in enumerate(dataloader[phase]):
                    src, tgt, dataset_id = data
                    src, tgt,dataset_id  = src.to(device), tgt.to(device), dataset_id.to(device)
                    batch = Batch(src, tgt, dataset_id)

                    binary_loss = run_once(model, batch, loss_function, model_opt, is_training)
                    total_loss += binary_loss

                    if i % args.tqdm_print_step == 0:
                        pbar.update(args.tqdm_print_step)
                        pbar.set_description(f'train loss: {binary_loss:.4f}')

            epoch_loss = total_loss / len(dataloader[phase])
            print(f"epoch loss: {epoch_loss}")

            if phase == 'val' and (epoch_loss < best_loss or epoch % 10 == 0) and epoch > 5:
                print("save model")
                best_loss = epoch_loss
                torch.save(model.state_dict(), f'{args.model_save_path}/epoch{epoch}_loss_{epoch_loss}.pkl')

if __name__ == "__main__":
    train_model()
