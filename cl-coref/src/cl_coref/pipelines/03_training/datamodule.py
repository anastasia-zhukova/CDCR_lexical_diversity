"""
Custom DataModule for training with clustering-based validation.
"""

import lightning as L
from torch.utils.data import DataLoader
import pandas as pd

from .validation_wrapper import ValidationDatasetWrapper


class CorefDataModule(L.LightningDataModule):
    """
    DataModule for coreference training with clustering-based validation.
    """
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_mentions_df: pd.DataFrame,
        dataset_name: str
    ):
        """
        Args:
            train_loader: DataLoader for training pairs
            val_loader: DataLoader for validation mentions (for encoding)
            val_mentions_df: DataFrame with validation mentions metadata
            dataset_name: Name of the dataset
        """
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader
        self.val_mentions_df = val_mentions_df
        self.dataset_name = dataset_name
        self.val_dataset_wrapper = None

    def setup(self, stage: str = None):
        """Setup validation dataset wrapper."""
        if stage == "fit" or stage is None:
            # Create validation dataset wrapper for clustering
            self.val_dataset_wrapper = ValidationDatasetWrapper(
                self.val_mentions_df,
                self.dataset_name
            )

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader
