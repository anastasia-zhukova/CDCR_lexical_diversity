"""
Custom Lightning module with clustering-based validation.
"""

import torch
import lightning as L
import sys
from pathlib import Path

# Add CL_COREF to path
CL_COREF_PATH = Path(__file__).parents[5] / "CL_COREF"
sys.path.insert(0, str(CL_COREF_PATH))

from model import CustomContrastiveLoss, create_optimizer
from evaluation import return_embeddings_by_topic, predict_clusters, compute_cdcr_metrics


class CorefLightningModule(L.LightningModule):
    """
    Lightning module for siamese network training with clustering-based validation.
    """
    def __init__(self, model, config: dict):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.config = config
        self.model_config = config.get("model", config)  # Handle both nested and flat config

        self.loss_fn = CustomContrastiveLoss(
            margin=self.model_config.get("margin", 0.7)
        )

        self.validation_step_outputs = []
        self.clustering_threshold = self.model_config.get("clustering_threshold", 0.5)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids1, attention_masks1, input_ids2, attention_masks2, labels = batch

        output1 = self(input_ids1, attention_masks1)
        output2 = self(input_ids2, attention_masks2)

        loss = self.loss_fn(output1, output2, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Encode validation mentions and store their embeddings."""
        indices, input_ids, attention_masks = batch
        embeddings = self(input_ids, attention_masks)

        self.validation_step_outputs.append({
            "indices": indices.cpu(),
            "embeddings": embeddings.cpu()
        })
        return embeddings

    def on_validation_epoch_end(self):
        """Perform clustering-based validation using all validation mentions."""
        if not self.validation_step_outputs:
            return

        # Get validation dataset wrapper from datamodule
        if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
            print("Warning: No datamodule found, skipping validation clustering")
            self.validation_step_outputs.clear()
            return

        val_dataset_wrapper = self.trainer.datamodule.val_dataset_wrapper
        if val_dataset_wrapper is None:
            print("Warning: No validation dataset wrapper, skipping validation clustering")
            self.validation_step_outputs.clear()
            return

        # Concatenate all embeddings and indices
        all_indices = torch.cat([x["indices"] for x in self.validation_step_outputs])
        all_embeddings = torch.cat([x["embeddings"] for x in self.validation_step_outputs])

        # Create mention_id to embedding mapping
        val_mentions_df = self.trainer.datamodule.val_mentions_df
        dict_embeddings_by_mention = {}

        for idx, emb in zip(all_indices.tolist(), all_embeddings):
            mention_id = val_mentions_df.iloc[idx]['mention_id']
            dict_embeddings_by_mention[mention_id] = emb.to(self.device)

        # Group embeddings by topic
        dict_embeddings_by_topic = return_embeddings_by_topic(
            dict_embeddings_by_mention,
            val_dataset_wrapper.mentions,
            use_dataset_prefix=False  # Single dataset scenario
        )

        # Predict clusters using hierarchical clustering
        predicted_clusters_df = predict_clusters(
            dict_embeddings_by_topic,
            clustering_threshold=self.clustering_threshold
        )

        # Merge with gold clusters
        merged_df = val_dataset_wrapper.gold_clusters_df.merge(
            predicted_clusters_df,
            how="inner",
            on="mention_id"
        )

        # Compute metrics
        df_metrics = compute_cdcr_metrics(
            cdcr_dataset_name=val_dataset_wrapper.name,
            predictions_df=merged_df,
            metrics="all",
            mention_type="all"
        )

        # Log CoNLL F1
        f1_conll = df_metrics.loc["conll", "F1"]
        self.log("val_f1_conll", f1_conll, prog_bar=True)

        # Log all metrics
        for metric_name in df_metrics.index:
            if metric_name != "conll":  # Already logged conll as main metric
                self.log(f"val_{metric_name}_f1", df_metrics.loc[metric_name, "F1"], prog_bar=False)
                self.log(f"val_{metric_name}_precision", df_metrics.loc[metric_name, "Precision"], prog_bar=False)
                self.log(f"val_{metric_name}_recall", df_metrics.loc[metric_name, "Recall"], prog_bar=False)

        print(f"\nValidation Epoch {self.current_epoch} - CoNLL F1: {f1_conll:.2f}")
        print(f"Full metrics:\n{df_metrics}")

        # Clear outputs for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return create_optimizer(self.model_config["learning_rate"], self.model)
