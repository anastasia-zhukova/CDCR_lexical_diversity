import os
import yaml
import lightning as L
from torch.utils.data import DataLoader
from dataset import CL_Dataset
from dataloader import create_data_loader
from model import create_tokenizer

class CorefDataModule(L.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.data_config = config["data"]
        self.paths_config = config["paths"]

        self.special_mention_tokens = [f"[{t}]" for t in self.data_config["special_tokens"]] + \
                                     [f"[/{t}]" for t in self.data_config["special_tokens"]]

        # Load data setups configuration
        data_setups_path = os.path.join(os.path.dirname(__file__), "configs", "data_setups.yaml")
        with open(data_setups_path, "r") as f:
            self.data_setups = yaml.safe_load(f)

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

        # Get the data setup configuration
        data_setup_id = self.data_config["data_setup"]
        if data_setup_id not in self.data_setups:
            raise ValueError(f"Invalid data_setup: {data_setup_id}. Available setups: {list(self.data_setups.keys())}")

        setup_config = self.data_setups[data_setup_id]
        setup_name = setup_config["name"]

        # Build dataset configs for train and val splits
        train_dataset_configs = []
        val_dataset_configs = []

        for dataset_spec in setup_config["datasets"]:
            dataset_name = dataset_spec["name"]
            topic_filter = dataset_spec.get("topic_filter")
            max_mentions = dataset_spec.get("max_mentions")

            for split in dataset_spec["splits"]:
                config = {
                    "name": dataset_name,
                    "path": os.path.join(base_data_path, dataset_name, split),
                    "topic_filter": topic_filter,
                    "max_mentions": max_mentions,
                    "predicted_topics": self.data_config["predicted_topics"]
                }

                if split == "train":
                    train_dataset_configs.append(config)
                elif split == "val":
                    val_dataset_configs.append(config)

        if stage == "fit" or stage is None:
            self.train_dataset = CL_Dataset(
                name=f"{setup_name}_train",
                dataset_configs=train_dataset_configs,
                context_method=self.data_config["context_method"],
                tokenizer=self.tokenizer,
                special_mention_tokens=self.special_mention_tokens,
                mode="train",
                additional_sampling=self.data_config["additional_sampling"],
                sentence_transformer_model=self.config["model"]["sentence_transformer_model"]
            )

            self.val_dataset = CL_Dataset(
                name=f"{setup_name}_val",
                dataset_configs=val_dataset_configs,
                context_method=self.data_config["context_method"],
                tokenizer=self.tokenizer,
                special_mention_tokens=self.special_mention_tokens,
                mode="inference"
            )

    def train_dataloader(self):
        return create_data_loader(self.train_dataset, batch_size=self.data_config["train_batch_size"])

    def val_dataloader(self):
        return create_data_loader(self.val_dataset, batch_size=self.data_config["val_batch_size"])
