import argparse
import pyhocon
from tqdm import tqdm
from itertools import combinations
from sklearn.utils import shuffle
from torch.utils import data

from data_loader import MultiTaskBatchSampler
from models import FullCrossEncoder
from evaluator import Evaluation
from dataset import CrossEncoderDatasetFull, CrossEncoderMixedDatasetFull, ParaphraseDataset
from utils import *

if __name__ == '__main__':
    #Read config_paiwise.json file and config_datasets.json file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_pairwise.json')
    parser.add_argument('--dataset_config', type=str, default='configs/config_datasets.json')
    parser.add_argument('--config_experiments', type=str, default='configs/config_experiments.json')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    datasets_config = pyhocon.ConfigFactory.parse_file(args.dataset_config)
    datasets_config = resolve_data_path(datasets_config)
    config_experiments = pyhocon.ConfigFactory.parse_file(args.config_experiments)

    #Initialize logger and model path
    fix_seed(config)
    logger = create_logger(config, create_file=True)
    logger.setLevel(logging.INFO)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config.model_path)

    # init train and dev set
    if "Experiment" in config.experiment_id:
        train = CrossEncoderMixedDatasetFull(config,datasets_config,config_experiments[config.experiment_id]['train'])
        val = CrossEncoderMixedDatasetFull(config,datasets_config,config_experiments[config.experiment_id]['val'])
        #test = CrossEncoderMixedDatasetFull(config,datasets_config,config_experiments[config.experiment_id]['test'])

    else:
        if config.data_folder == "mrpc" or "HoW_DatasetDict" in config.data_folder:
            train = ParaphraseDataset(config, 'train') 
            val = ParaphraseDataset(config, 'validation')
        else:
            train = CrossEncoderDatasetFull(config, 'train') 
            val = CrossEncoderDatasetFull(config, 'val')

    #Create custom sampler
    sampler = MultiTaskBatchSampler(        dataset=train,
        batch_size=config.batch_size,
        total_epochs=config.epochs,
        sampling = config.sampling)
    
    #Create data_loaders
    train_loader = data.DataLoader(train,batch_sampler=sampler,pin_memory=True)
    val_loader = data.DataLoader(val, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    device_ids = config.gpu_num
    device = torch.device("cuda:{}".format(device_ids[0]))

    ## Models' initiation
    logger.info('Init models')
    cross_encoder_single = FullCrossEncoder(config).to(device)
    cross_encoder = torch.nn.DataParallel(cross_encoder_single, device_ids=device_ids)

    # Define the names of the layers to freeze
    unfreeze_layers = [
        "encoder.layer.21.output.dense.weight",
        "encoder.layer.21.output.dense.bias",
        "encoder.layer.21.output.LayerNorm.weight",
        "encoder.layer.21.output.LayerNorm.bias",

        "encoder.layer.22.output.dense.weight",
        "encoder.layer.22.output.dense.bias",
        "encoder.layer.22.output.LayerNorm.weight",
        "encoder.layer.22.output.LayerNorm.bias",

        "encoder.layer.23.output.dense.weight",
        "encoder.layer.23.output.dense.bias",
        "encoder.layer.23.output.LayerNorm.weight",
        "encoder.layer.23.output.LayerNorm.bias",
    ]
    #Unfreezes Roberta layers in list
    for name, param in cross_encoder_single.model.named_parameters():
        param.requires_grad = any(layer_name in name for layer_name in unfreeze_layers)

    # Unfreeze the parameters of the linear layers
    for param in cross_encoder_single.linear_cdcr.parameters():
        param.requires_grad = True

    # Unfreeze the parameters of the linear layers
    for param in cross_encoder_single.linear_pi.parameters():
        param.requires_grad = True

    ## Optimizer and loss function
    criterion = get_loss_function(config)
    optimizer = get_optimizer(config, [cross_encoder])
    scheduler = get_scheduler(optimizer, total_steps=config.epochs * len(train_loader))

    logger.info('Number of parameters of the pairwise classifier: {}'.format(
        count_parameters(cross_encoder)))

    ##################################################################################
    ####                    TRAINING
    ##################################################################################

    f1 = []
    best_f1 = 0.0

    for epoch in range(config.epochs):
        logger.info('Epoch: {}'.format(epoch))
        accumulate_loss = 0
        number_of_positive_pairs, number_of_pairs = 0, 0
        cross_encoder.train()

        for batch_x, batch_y, instance_id, dataset_task in tqdm(train_loader):
            #Separate sentences from batch to add the tokenizers special tokens
            before_sep = [parts[0] for parts in (string.split("[SEP]") for string in batch_x)]
            after_sep = [parts[1] for parts in (string.split("[SEP]") for string in batch_x)]

            bert_tokens = cross_encoder_single.tokenizer(text=before_sep,text_pair=after_sep,add_special_tokens=True,padding=True,truncation=True)
            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)

            optimizer.zero_grad()
            cross_encoder.zero_grad()
            scores = cross_encoder(input_ids, attention_mask,dataset_task)

            batch_y = batch_y.view(-1, 1)
            loss = criterion(scores, batch_y.to(device))
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(cross_encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            torch.cuda.empty_cache()


            accumulate_loss += loss.item()

            number_of_positive_pairs += len((batch_y == 1).nonzero())
            number_of_pairs += len(batch_y)

        # list_of_topics = shuffle(list(range(len(training_set.topic_list))))
        # number_of_positive_pairs = 0
        # number_of_pairs = 0
        # for topic_num in tqdm(list_of_topics):
        #
        #
        #
        #
        #     topic = training_set.topic_list[topic_num]
        #     first, second, labels = get_candidate_spans(training_set, topic)
        #
        #     number_of_positive_pairs += len((labels == 1).nonzero())
        #     number_of_pairs += len(labels)
        #
        #
        #     pairwises = CrossEncoderDataset(training_set.mentions_repr, first, second, labels)
        #     generator = data.DataLoader(pairwises, shuffle=True, batch_size=config.batch_size)
        #     loss = fine_tune_bert(cross_encoder, generator, criterion, optimizer, device)
        #     accumulate_loss += loss


        logger.info('Number of positive/total pairs: {}/{}'.format(number_of_positive_pairs, number_of_pairs))
        logger.info('Accumulate loss: {}'.format(accumulate_loss))


        logger.info('Evaluate on the dev set')
        all_scores, all_labels = [], []
        number_of_positive_pairs, number_of_pairs = 0, 0
        cross_encoder.eval()

        for batch_x, batch_y,instance_id,dataset_task in tqdm(val_loader):
            bert_tokens = cross_encoder_single.tokenizer(batch_x, pad_to_max_length=True)
            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)

            with torch.no_grad():
                scores = cross_encoder(input_ids, attention_mask,dataset_task)

            torch.cuda.empty_cache()
            
            number_of_positive_pairs += len((batch_y == 1).nonzero())
            number_of_pairs += len(batch_y)


        # for topic_num, topic in enumerate(tqdm(dev_set.topic_list)):
        #     dev_topic = dev_set.topic_list[topic_num]
        #     first, second, labels = get_candidate_spans(dev_set, dev_topic)
        #     number_of_positive_pairs += len((labels == 1).nonzero())
        #     number_of_pairs += len(labels)
        #
        #     pairwises = CrossEncoderDataset(dev_set.mentions_repr, first, second, labels)
        #     generator = data.DataLoader(pairwises, shuffle=True, batch_size=config.batch_size)
        #     scores, labels = evaluate_model(cross_encoder, generator, device)
        #     torch.cuda.empty_cache()
        #
            all_scores.extend(scores.squeeze(1))
            all_labels.extend(batch_y.squeeze(1))



        all_labels = torch.stack(all_labels)
        all_scores = torch.stack(all_scores)


        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        logger.info('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                             len(all_labels)))

        logger.info('Min score: {}'.format(all_scores.min().item()))
        logger.info('Max score: {}'.format(all_scores.max().item()))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}, Accuracy: {}'.
                    format(eval.get_recall(), eval.get_precision(), eval.get_f1(), eval.get_accuracy()))


        out_dir = os.path.join(config.model_path, f"{config.experiment_id}",f"{'base' if 'base' in config.roberta_model else 'large'}_{config.batch_size}",
                      f"{config.sampling}_sampling",f"checkpoint_{epoch}")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        current_f1 = eval.get_f1() 
        if current_f1 > best_f1:
            best_f1 = current_f1
            cross_encoder_single.model.save_pretrained(os.path.join(out_dir, 'bert'))
            torch.save(cross_encoder_single.linear_cdcr.state_dict(), os.path.join(out_dir, 'linear_cdcr'))
            torch.save(cross_encoder_single.linear_cdcr.state_dict(), os.path.join(out_dir, 'linear_pi'))

        f1.append(eval.get_f1())
    print("F1 values",f1)
