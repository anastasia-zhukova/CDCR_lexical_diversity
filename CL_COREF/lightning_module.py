import torch
import lightning as L
from model import create_siamese_model, CustomContrastiveLoss, create_optimizer
from evaluation import predict_and_evaluate

class CorefLightningModule(L.LightningModule):
    def __init__(self, config: dict, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.config = config
        self.model_config = config["model"]
        self.tokenizer = tokenizer
        
        self.model = create_siamese_model(self.model_config["language_model"], self.tokenizer)
        self.loss_fn = CustomContrastiveLoss(margin=self.model_config["margin"])
        
        self.validation_step_outputs = []

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
        # The validation batch is (index, input_ids, attention_mask) from create_data_loader for inference mode
        # Wait, the current test_func in train.py calls predict_and_evaluate which takes the whole dataloader.
        # To make it truly PL-compliant, we should probably collect embeddings in validation_step
        # and then evaluate in on_validation_epoch_end.
        
        indices, input_ids, attention_masks = batch
        embeddings = self(input_ids, attention_masks)
        
        self.validation_step_outputs.append({
            "indices": indices.cpu(),
            "embeddings": embeddings.cpu()
        })
        return embeddings

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # Concatenate all embeddings and indices
        all_indices = torch.cat([x["indices"] for x in self.validation_step_outputs])
        all_embeddings = torch.cat([x["embeddings"] for x in self.validation_step_outputs])
        
        # We need the cl_cdcr_dataset to get the keys and gold labels
        # This is slightly tricky with PL because the dataset is in the DataModule.
        # However, we can access the dataloader's dataset if needed, or pass it in.
        # For now, let's assume we can re-use the evaluation logic by reconstructing the dict.
        
        datamodule = self.trainer.datamodule
        val_dataset = datamodule.val_dataset
        
        dict_embeddings_by_mention = {}
        for idx, emb in zip(all_indices.tolist(), all_embeddings):
            key = val_dataset.keys_list[idx]
            dict_embeddings_by_mention[key] = emb.to(self.device)

        # Call the existing evaluation logic
        from evaluation import return_embeddings_by_topic, predict_clusters, compute_cdcr_metrics
        
        cdcr_dataset = val_dataset.cdcr_dataset
        dict_embeddings_by_topic = return_embeddings_by_topic(dict_embeddings_by_mention, cdcr_dataset.mentions)
        predicted_clusters_df = predict_clusters(dict_embeddings_by_topic, self.model_config["clustering_threshold"])
        merged_df = cdcr_dataset.gold_clusters_df.merge(predicted_clusters_df, how="inner", on="mention_id")
        
        df_metrics = compute_cdcr_metrics(
            cdcr_dataset_name=cdcr_dataset.name,
            predictions_df=merged_df,
            metrics="all",
            mention_type=val_dataset.mention_type
        )
        
        f1_conll = df_metrics.loc["conll", "F1"]
        self.log("val_f1_conll", f1_conll, prog_bar=True)
        
        print(f"Validation F1 CoNLL: {f1_conll:.2f}")
        
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return create_optimizer(self.model_config["learning_rate"], self.model)
