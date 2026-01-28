import os
import lightning as L
from torch.utils.data import DataLoader
from .dataset import CL_Dataset
from .dataloader import create_data_loader
from .model import create_tokenizer

class CorefDataModule(L.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.data_config = config["data"]
        self.paths_config = config["paths"]
        
        self.special_mention_tokens = [f"[{t}]" for t in self.data_config["special_tokens"]] + \
                                     [f"[/{t}]" for t in self.data_config["special_tokens"]]
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # Download data if necessary (already present in this case)
        pass

    def setup(self, stage: str = None):
        if self.tokenizer is None:
            self.tokenizer = create_tokenizer(
                self.config["model"]["language_model"], 
                self.special_mention_tokens
            )

        base_data_path = os.path.abspath(self.paths_config["base_data_path"])
        
        # Setup data paths based on data_setup
        if self.data_config["data_setup"] == 0:
            train_paths = [os.path.join(base_data_path, "ECBplus-prep", "output_data", 'train')]
            val_paths = [os.path.join(base_data_path, "ECBplus-prep", "output_data", 'val')]
            train_name, val_name = "ecbplus_train", "ecbplus_val"
        elif self.data_config["data_setup"] == 1:
            train_paths = [
                os.path.join(base_data_path, "ECBplus-prep", "output_data", 'train'),
                os.path.join(base_data_path, "FCC-prep", "output_data_FCC-T", 'train')
            ]
            val_paths = [
                os.path.join(base_data_path, "ECBplus-prep", "output_data", 'val'),
                os.path.join(base_data_path, "FCC-prep", "output_data_FCC-T", 'val')
            ]
            train_name, val_name = "ecbplus_fcc_train", "ecbplus_fcc_val"
        elif self.data_config["data_setup"] == 2:
            train_paths = [
                os.path.join(base_data_path, "ECBplus-prep", "output_data", 'train'),
                os.path.join(base_data_path, "NP4E-prep", "output_data", 'train')
            ]
            val_paths = [
                os.path.join(base_data_path, "ECBplus-prep", "output_data", 'val'),
                os.path.join(base_data_path, "NP4E-prep", "output_data", 'val')
            ]
            train_name, val_name = "ecbplus_np4e_train", "ecbplus_np4e_val"
        else:
            raise ValueError(f"Invalid data_setup: {self.data_config['data_setup']}")

        if stage == "fit" or stage is None:
            self.train_dataset = CL_Dataset(
                name=train_name,
                data_paths=train_paths,
                context_method=self.data_config["context_method"],
                mention_type=self.data_config["mention_type"],
                tokenizer=self.tokenizer,
                special_mention_tokens=self.special_mention_tokens,
                mode="train",
                additional_sampling=self.data_config["additional_sampling"],
                predicted_topics=self.data_config["predicted_topics"]
            )
            
            self.val_dataset = CL_Dataset(
                name=val_name,
                data_paths=val_paths,
                context_method=self.data_config["context_method"],
                mention_type=self.data_config["mention_type"],
                tokenizer=self.tokenizer,
                special_mention_tokens=self.special_mention_tokens,
                mode="inference",
                predicted_topics=self.data_config["predicted_topics"]
            )

    def train_dataloader(self):
        return create_data_loader(self.train_dataset, batch_size=self.data_config["train_batch_size"])

    def val_dataloader(self):
        return create_data_loader(self.val_dataset, batch_size=self.data_config["val_batch_size"])
