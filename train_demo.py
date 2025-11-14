from transformers import BertTokenizer
from util.data_loader import get_data_loader
from util.framework import FewShotNERExecutor
from util.word_encoder import BERTWordEncoder
from model.proto import PrototypicalNetwork
from model.nnshot import NearestNeighborShot
from model.container import ContainerModel
from model.proml import ProMLModel
from model.supervised import TransferBERTModel
import torch
from torch import optim, nn
import numpy as np
import argparse
import os
import torch
import random


def initialize_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', default='intra',
                        help='training mode, must be in [inter, intra, supervised]')
    parser.add_argument('--train_way', default=5, type=int,
                        help='N in train')
    parser.add_argument('--way', default=5, type=int,
                        help='N way')
    parser.add_argument('--train_shot', default=1, type=int,
                        help='K in train')
    parser.add_argument('--shot', default=1, type=int,
                        help='K shot')
    parser.add_argument('--query_num', default=1, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--training_iterations', default=600, type=int,
                        help='num of iters in training')
    parser.add_argument('--validation_iterations', default=100, type=int,
                        help='num of iters in validation')
    parser.add_argument('--testing_iterations', default=500, type=int,
                        help='num of iters in testing')
    parser.add_argument('--validation_interval', default=20, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model_type', default='proto',
                        help='model name, must be basic-bert, proto, nnshot, or structshot')
    parser.add_argument('--sequence_max_length', default=100, type=int,
                        help='max length')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--load_checkpoint', default=None,
                        help='load ckpt')
    parser.add_argument('--save_checkpoint', default=None,
                        help='save ckpt')
    parser.add_argument('--use_fp16', action='store_true',
                        help='use nvidia apex fp16')
    parser.add_argument('--test_only', action='store_true',
                        help='only test')
    parser.add_argument('--checkpoint_name', type=str, default='',
                        help='checkpoint name.')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--ignore_label_index', type=int, default=-1,
                        help='label index to ignore when calculating loss and metrics')
    parser.add_argument('--use_sampled_dataset', action='store_true',
                        help='use released sampled data, the data should be stored at "data/episode-data/" ')

    parser.add_argument('--pretrained_checkpoint', default=None,
                        help='bert / roberta pre-trained checkpoint')

    parser.add_argument('--use_dot_product', action='store_true',
                        help='use dot instead of L2 distance for proto')

    parser.add_argument('--temperature', default=0.05, type=float,
                        help='StructShot parameter to re-normalizes the transition probabilities')

    parser.add_argument('--use_sgd_for_bert', action='store_true',
                        help='use SGD instead of AdamW for BERT.')

    parser.add_argument('--projection_dimension', type=int, default=32, help='the dimension of gaussian embedding')
    parser.add_argument('--mask_probability', type=float)

    parser.add_argument('--evaluate_with_finetune', action='store_true', help='finetune on support set')
    parser.add_argument('--enable_visualization', action='store_true')
    parser.add_argument('--use_ontology_split', type=str, choices=['A', 'B', 'C'], help='flag for OntoNotesABC splits')
    parser.add_argument('--use_wnut_dataset', action='store_true', help='flag for WNUT')
    parser.add_argument('--use_ontonotes_dataset', action='store_true', help='flag for OntoNotes')
    parser.add_argument('--use_conll2003_dataset', action='store_true', help='flag for CoNLL 2003')
    parser.add_argument('--use_i2b2_dataset', action='store_true', help='flag for I2B2')
    parser.add_argument('--use_gum_dataset', action='store_true', help='flag for GUM')
    parser.add_argument('--full_test_mode', action='store_true', help='run test in low-resource evaluation mode')
    parser.add_argument('--total_way_count', type=int, default=5,
                        help='total N in support set used in low-resource evaluation')
    parser.add_argument('--mixing_ratio', type=float, default=0.5,
                        help='the weighted averaging hyperparameter for ProML')
    parser.add_argument('--evaluation_mixing_ratio', type=float,
                        help='the weighted averaging hyperparameter for ProML used in evalution')
    parser.add_argument('--top_neighbors', type=int, default=1, help='KNN in inference')
    parser.add_argument('--result_output_file', type=str,
                        help='write inference results to file, only for low-resource evaluation')
    parser.add_argument('--disable_shuffling', action='store_true')
    parser.add_argument('--training_classes_count', type=int, default=50, help='used for transferBERT baseline')
    parser.add_argument('--validation_classes_count', type=int, default=50, help='used for transferBERT baseline')
    parser.add_argument('--testing_classes_count', type=int, default=50, help='used for transferBERT baseline')
    parser.add_argument('--sample_support_to_directory', type=str,
                        help='only sample support set with a specified directory')
    parser.add_argument('--support_set_path', type=str)
    parser.add_argument('--query_set_path', type=str)
    parser.add_argument('--no_separator', action='store_true', help='no separator in prompts')

    parser.add_argument('--experiment_name', type=str, required=True)

    args = parser.parse_args()
    train_way = args.train_way
    way = args.way
    train_shot = args.train_shot
    shot = args.shot
    query_num = args.query_num
    batch_size = args.batch_size
    model_type = args.model_type
    sequence_max_length = args.sequence_max_length


    initialize_random_seed(args.random_seed)
    pretrained_checkpoint = args.pretrained_checkpoint or 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    if args.use_ontology_split is not None:
        args.training_data = 'data/ontoNotes/__train_{}.txt'.format(args.use_ontology_split)
        args.testing_data = 'data/ontoNotes/__test_{}.txt'.format(args.use_ontology_split)
        args.validation_data = 'data/ontoNotes/__dev_{}.txt'.format(args.use_ontology_split)
        dataset_name = 'OntoNotes_{}'.format(args.use_ontology_split)

    elif args.use_ontonotes_dataset:
        args.training_data = 'data/ontoNotes/train.txt'
        args.testing_data = 'data/ontoNotes/test.txt'
        args.validation_data = 'data/ontoNotes/dev.txt'
        dataset_name = 'OntoNotes'

    elif args.use_wnut_dataset:
        args.training_data = 'data/ontoNotes/train.txt'
        args.validation_data = 'data/wnut-dev.txt'
        args.testing_data = 'data/wnut-test.txt'
        dataset_name = 'WNUT'

    elif args.use_conll2003_dataset:
        args.training_data = 'data/ontoNotes/train.txt'
        args.validation_data = 'data/conll-dev.txt'
        args.testing_data = 'data/conll-test.txt'
        dataset_name = 'CoNLL2003'

    elif args.use_i2b2_dataset:
        args.training_data = 'data/ontoNotes/train.txt'
        args.validation_data = 'data/i2b2-test.txt'
        args.testing_data = 'data/i2b2-test.txt'
        dataset_name = 'I2B2'

    elif args.use_gum_dataset:
        args.training_data = 'data/ontoNotes/train.txt'
        args.validation_data = 'data/gum-test.txt'
        args.testing_data = 'data/gum-test.txt'
        dataset_name = 'GUM'

    elif not args.use_sampled_dataset:
        args.training_data = f'data/{args.training_mode}/train.txt'
        args.testing_data = f'data/{args.training_mode}/test.txt'
        args.validation_data = f'data/{args.training_mode}/dev.txt'
        if not (os.path.exists(args.training_data) and os.path.exists(args.validation_data) and os.path.exists(
                args.testing_data)):
            os.system(f'bash data/download.sh {args.training_mode}')
    else:
        args.training_data = f'data/episode-data/{args.training_mode}/train_{args.way}_{args.shot}.jsonl'
        args.testing_data = f'data/episode-data/{args.training_mode}/test_{args.way}_{args.shot}.jsonl'
        args.validation_data = f'data/episode-data/{args.training_mode}/dev_{args.way}_{args.shot}.jsonl'
        if not (os.path.exists(args.training_data) and os.path.exists(args.validation_data) and os.path.exists(
                args.testing_data)):
            os.system(f'bash data/download.sh episode-data')
            os.system('unzip -d data/ data/episode-data.zip')

    if not args.full_test_mode:
        training_data_loader = get_data_loader(args.training_data, tokenizer,
                                               N=train_way, K=train_shot, Q=query_num, batch_size=batch_size,
                                               max_length=sequence_max_length, ignore_index=args.ignore_label_index,
                                               use_sampled_data=args.use_sampled_dataset,
                                               no_shuffle=args.disable_shuffling, no_sep=args.no_separator)
        validation_data_loader = get_data_loader(args.validation_data, tokenizer,
                                                 N=way, K=shot, Q=query_num, batch_size=batch_size,
                                                 max_length=sequence_max_length, ignore_index=args.ignore_label_index,
                                                 use_sampled_data=args.use_sampled_dataset,
                                                 no_shuffle=args.disable_shuffling, no_sep=args.no_separator)
        testing_data_loader = get_data_loader(args.testing_data, tokenizer,
                                              N=way, K=shot, Q=query_num, batch_size=batch_size,
                                              max_length=sequence_max_length, ignore_index=args.ignore_label_index,
                                              use_sampled_data=args.use_sampled_dataset,
                                              no_shuffle=args.disable_shuffling, no_sep=args.no_separator)
    else:
        training_data_loader = validation_data_loader = None
        extra_data_loader = get_data_loader(args.support_set_path or args.testing_data, tokenizer,
                                            N=args.total_way_count, K=shot, Q=query_num, batch_size=1,
                                            max_length=sequence_max_length, ignore_index=args.ignore_label_index,
                                            use_sampled_data=args.use_sampled_dataset,
                                            i2b2flag=args.use_i2b2_dataset or args.use_gum_dataset,
                                            dataset_name=dataset_name, no_shuffle=args.disable_shuffling, is_extra=True,
                                            no_sep=args.no_separator)

        testing_data_loader_creator, testing_data_set = get_data_loader(args.query_set_path or args.testing_data,
                                                                        tokenizer,
                                                                        N=way, K=shot, Q=query_num, batch_size=1,
                                                                        max_length=sequence_max_length,
                                                                        ignore_index=args.ignore_label_index,
                                                                        use_sampled_data=args.use_sampled_dataset,
                                                                        full_test=True,
                                                                        no_shuffle=args.disable_shuffling,
                                                                        no_sep=args.no_separator)
        testing_data_loader = testing_data_loader_creator, testing_data_set

    if args.sample_support_to_directory is not None:
        os.makedirs(args.sample_support_to_directory, exist_ok=True)
        assert args.full_test_mode
        for _, (support, not_used) in enumerate(extra_data_loader):
            if _ >= 10:
                break

            with open(os.path.join(args.sample_support_to_directory, '{}.txt'.format(_)), 'w') as f:
                f.write('\n\n'.join([testing_data_set.samples[index].__str__() for index in support['index']]))
                f.write('\n\n')

        return

    word_encoder = BERTWordEncoder(
        pretrained_checkpoint, tokenizer)

    prefix = args.experiment_name

    if model_type == 'proto':
        model = PrototypicalNetwork(word_encoder, dot=args.use_dot_product, ignore_index=args.ignore_label_index)
        executor = FewShotNERExecutor(training_data_loader, validation_data_loader, testing_data_loader,
                                      use_sampled_data=args.use_sampled_dataset,
                                      extra_data_loader=extra_data_loader if args.full_test_mode else None)
    elif model_type == 'nnshot':
        model = NearestNeighborShot(word_encoder, dot=args.use_dot_product,
                                    ignore_index=args.ignore_label_index)
        executor = FewShotNERExecutor(training_data_loader, validation_data_loader, testing_data_loader,
                                      use_sampled_data=args.use_sampled_dataset,
                                      extra_data_loader=extra_data_loader if args.full_test_mode else None)
    elif model_type == 'structshot':
        model = NearestNeighborShot(word_encoder, dot=args.use_dot_product,
                                    ignore_index=args.ignore_label_index)
        executor = FewShotNERExecutor(training_data_loader, validation_data_loader, testing_data_loader, N=args.way,
                                      tau=args.temperature, train_fname=args.training_data,
                                      viterbi=True, use_sampled_data=args.use_sampled_dataset,
                                      extra_data_loader=extra_data_loader if args.full_test_mode else None)
    elif model_type == 'container':
        print('use container')
        model = ContainerModel(word_encoder, dot=args.use_dot_product, ignore_index=args.ignore_label_index,
                               gaussian_dim=args.projection_dimension)
        executor = FewShotNERExecutor(training_data_loader, validation_data_loader, testing_data_loader, N=args.way,
                                      tau=args.temperature, train_fname=args.training_data,
                                      viterbi=False, use_sampled_data=args.use_sampled_dataset, contrast=True,
                                      extra_data_loader=extra_data_loader if args.full_test_mode else None)

    elif model_type == 'ProML':
        print('use ProML')
        model = ProMLModel(word_encoder, dot=args.use_dot_product, ignore_index=args.ignore_label_index,
                           proj_dim=args.projection_dimension, mix_rate=args.mixing_ratio, topk=args.top_neighbors)
        executor = FewShotNERExecutor(training_data_loader, validation_data_loader, testing_data_loader, N=args.way,
                                      tau=args.temperature, train_fname=args.training_data, viterbi=False,
                                      use_sampled_data=args.use_sampled_dataset, contrast=True,
                                      extra_data_loader=extra_data_loader if args.full_test_mode else None,
                                      eval_topk=args.top_neighbors, eval_mix_rate=args.evaluation_mixing_ratio)

    elif model_type == 'transfer':
        print('use transfer bert')
        model = TransferBERTModel(word_encoder, dot=args.use_dot_product, ignore_index=args.ignore_label_index,
                                  train_classes=args.training_classes_count, val_classes=args.validation_classes_count,
                                  test_classes=args.testing_classes_count)
        executor = FewShotNERExecutor(training_data_loader, validation_data_loader, testing_data_loader, N=args.way,
                                      use_sampled_data=args.use_sampled_dataset, contrast=False,
                                      extra_data_loader=extra_data_loader if args.full_test_mode else None)

    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    checkpoint_path = 'checkpoint/{}.pth.tar'.format(prefix)
    if args.save_checkpoint:
        checkpoint_path = args.save_checkpoint

    if torch.cuda.is_available():
        model.cuda()

    if not args.test_only:
        if args.learning_rate == -1:
            args.learning_rate = 2e-5

        executor.train(model, prefix,
                       load_ckpt=args.load_checkpoint, save_ckpt=checkpoint_path,
                       val_step=args.validation_interval, fp16=args.use_fp16,
                       train_iter=args.training_iterations, warmup_step=1000, val_iter=args.validation_iterations,
                       learning_rate=args.learning_rate, use_sgd_for_bert=args.use_sgd_for_bert,
                       mask_rate=args.mask_probability, mask_id=tokenizer.mask_token_id,
                       finetuning=True if args.model_type == 'transfer' else False)
    else:
        checkpoint_path = args.load_checkpoint
        if checkpoint_path is None:
            checkpoint_path = 'none'

    if args.enable_visualization:
        print(args.experiment_name)
        executor.visualize(model, 100, ckpt=checkpoint_path,
                           part='test', exp_name=args.experiment_name)

    precision, recall, f1, fp, fn, within, outer = executor.eval(model, args.testing_iterations, ckpt=checkpoint_path,
                                                                 finetuning=args.evaluate_with_finetune,
                                                                 finetuning_lr=3e-5, full_test=args.full_test_mode,
                                                                 finetuning_mix_rate=False,
                                                                 output_file=args.result_output_file)
    print("RESULT: precision: %.4f, recall: %.4f, f1:%.4f" %
          (precision, recall, f1))
    print('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within:%.4f, outer: %.4f' %
          (fp, fn, within, outer))


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')
    run_main()