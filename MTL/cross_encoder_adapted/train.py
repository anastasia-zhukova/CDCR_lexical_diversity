import argparse
import pyhocon
from tqdm import tqdm
from itertools import combinations
from sklearn.utils import shuffle
from torch.utils import data


from models import  SpanEmbedder, SpanScorer, FullCrossEncoder
from evaluator import Evaluation
from dataset import CrossEncoderDatasetFull
from utils import *
from cluster_documents import return_mentions_with_predicted_topics





if __name__ == '__main__':
    #Read config_paiwise.json file and config_datasets.json file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_pairwise.json')
    parser.add_argument('--dataset_config', type=str, default='configs/config_datasets.json')
    args = parser.parse_args()
    config = pyhocon.ConfigFactory.parse_file(args.config)
    datasets_config = pyhocon.ConfigFactory.parse_file(args.dataset_config)

    #Set data paths according to the common_data_folder path in the config_datasets file
    common_data_folder = datasets_config.get_string("common_data_folder")
    for key in datasets_config:
        if isinstance(datasets_config[key], dict) and "data_folder" in datasets_config[key]:
            datasets_config[key]["data_folder"] = datasets_config[key]["data_folder"].replace("${common_data_folder}", common_data_folder)

 

    #Initialize logger and model path
    fix_seed(config)
    logger = create_logger(config, create_file=True)
    logger.setLevel(logging.INFO)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config.model_path)

    #Load config_datasets.json file
    datasets_config = pyhocon.ConfigFactory.parse_file('configs/config_datasets.json')
    common_data_folder = datasets_config.get_string("common_data_folder")
    for key in datasets_config:
        if isinstance(datasets_config[key], dict) and "data_folder" in datasets_config[key]:
            datasets_config[key]["data_folder"] = datasets_config[key]["data_folder"].replace("${common_data_folder}", common_data_folder)


    # init train and dev set
    train = CrossEncoderDatasetFull(config, 'train')
    train.create_mention_pairs_by_topics()
    train.negative_sampling()
    train.prepare_pair_of_mentions(mentions=train.mentions, first=train.first, second=train.second)

    train_loader = data.DataLoader(train, batch_size=config.batch_size, shuffle=True)
    dev = CrossEncoderDatasetFull(config, 'val')

    predict_topics = True #TODO Pass by config parameter
    if predict_topics:
        mentions_w_predicted_topics = return_mentions_with_predicted_topics(dev)
        dev.assign_mentions_with_new_topics(mentions_w_predicted_topics)
    #Else by default now: use the known topics at inference, which must change

    dev.create_mention_pairs_by_topics()
    dev.prepare_pair_of_mentions( mentions=dev.mentions, first=dev.first, second=dev.second)

    dev_loader = data.DataLoader(dev, batch_size=config.batch_size, shuffle=False)

    device_ids = config.gpu_num
    device = torch.device("cuda:{}".format(device_ids[0]))


    ## Models' initiation
    logger.info('Init models')
    # span_repr = SpanEmbedder(config, device).to(device)
    # span_scorer = SpanScorer(config).to(device)
    cross_encoder_single = FullCrossEncoder(config).to(device)
    cross_encoder = torch.nn.DataParallel(cross_encoder_single, device_ids=device_ids)


    # if config.training_method in ('pipeline', 'continue') and not config.use_gold_mentions:
    #     span_repr.load_state_dict(torch.load(config.span_repr_path, map_location=device))
    #     span_scorer.load_state_dict(torch.load(config.span_scorer_path, map_location=device))


    ## Optimizer and loss function
    criterion = get_loss_function(config)
    optimizer = get_optimizer(config, [cross_encoder])
    scheduler = get_scheduler(optimizer, total_steps=config.epochs * len(train_loader))


    # logger.info('Number of parameters of mention extractor: {}'.format(
    #     count_parameters(span_repr) + count_parameters(span_scorer)))
    logger.info('Number of parameters of the pairwise classifier: {}'.format(
        count_parameters(cross_encoder)))

    ##################################################################################
    ####                    TRAINING
    ##################################################################################



    logger.info('Number of topics: {}'.format(len(train.topics)))
    f1 = []
    for epoch in range(config.epochs):
        logger.info('Epoch: {}'.format(epoch))
        accumulate_loss = 0
        number_of_positive_pairs, number_of_pairs = 0, 0
        cross_encoder.train()

        for batch_x, batch_y in tqdm(train_loader):
            bert_tokens = cross_encoder.module.tokenizer(batch_x, pad_to_max_length=True)
            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)

            optimizer.zero_grad()
            cross_encoder.zero_grad()
            scores = cross_encoder(input_ids, attention_mask)
            loss = criterion(scores, batch_y.to(device))
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(cross_encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()


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

        for batch_x, batch_y in tqdm(dev_loader):
            bert_tokens = cross_encoder.module.tokenizer(batch_x, pad_to_max_length=True)
            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)

            with torch.no_grad():
                scores = cross_encoder(input_ids, attention_mask)


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


        out_dir = os.path.join(config.model_path, '{}_{}'.format(
            'base' if 'base' in config.roberta_model else 'large', config.batch_size),
            'checkpoint_{}'.format(epoch))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


        cross_encoder_single.model.save_pretrained(os.path.join(out_dir, 'bert'))
        torch.save(cross_encoder_single.linear.state_dict(), os.path.join(out_dir, 'linear'))

        f1.append(eval.get_f1())
