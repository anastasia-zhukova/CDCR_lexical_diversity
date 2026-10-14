import yaml
import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import wandb

from datamodule import CorefDataModule
from lightning_module import CorefLightningModule

def main(args):
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with CLI arguments if provided
    if args.mention_type:
        config["data"]["mention_type"] = args.mention_type
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.learning_rate:
        config["model"]["learning_rate"] = args.learning_rate
    if args.data_setup is not None:
        config["data"]["data_setup"] = args.data_setup
    if args.precision:
        config["training"]["precision"] = args.precision

    # Set random seed
    L.seed_everything(42)

    # Initialize DataModule
    datamodule = CorefDataModule(config)
    datamodule.setup() # Initialize tokenizer

    # Initialize LightningModule
    model = CorefLightningModule(config, datamodule.tokenizer)

    # Logging and Checkpointing
    tb_logger = TensorBoardLogger("logs", name=f"{config['data']['mention_type']}_{config['model']['language_model']}")

    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(
        project="cdcr-lexical-diversity",
        name=f"{config['data']['mention_type']}_{config['model']['language_model']}_setup{config['data']['data_setup']}",
        config=config,
        log_model=False
    )
    
    # Model path logic similar to old train.py
    out_dir = os.path.join(
        os.getcwd(), 
        config["paths"]["models_dir"], 
        config["data"]["mention_type"], 
        config["model"]["language_model"], 
        f"tokens_{'_'.join(config['data']['special_tokens'])}"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        filename=f"siamese_model_setup_{config['data']['data_setup']}",
        monitor="val_f1_conll",
        mode="max",
        save_top_k=1
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=config["training"]["num_epochs"],
        precision=config["training"]["precision"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        logger=[tb_logger, wandb_logger],
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0 # From old train.py clip_grad_norm_
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CL_COREF with PyTorch Lightning")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--mention_type", type=str, help="Override mention type")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--data_setup", type=int, help="Override data setup")
    parser.add_argument("--precision", type=str, help="Override precision (e.g., 16-mixed, 32)")
    
    args = parser.parse_args()
    main(args)