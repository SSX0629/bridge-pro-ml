import os
import numpy as np
import sys

from util.visualize import visualize
from . import word_encoder
from . import data_loader
import torch
import torch.distributions
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

from .viterbi import ViterbiDecoder


def calculate_abstract_transitions(training_filename, use_sampled_data=True):
    if use_sampled_data:
        samples = data_loader.FewShotNERDataset(training_filename, None, 1).samples
        tag_sequences = []
        for sample in samples:
            tag_sequences += sample['support']['label'] + sample['query']['label']
    else:
        samples = data_loader.FewShotNERDatasetWithRandomSampling(training_filename, None, 1, 1, 1, 1).samples
        tag_sequences = [sample.tags for sample in samples]

    start_outside, start_inside = 0., 0.
    outside_outside, outside_inside = 0., 0.
    inside_outside, inside_inside, cross_type = 0., 0., 0.
    for tags in tag_sequences:
        if tags[0] == 'O':
            start_outside += 1
        else:
            start_inside += 1
        for i in range(len(tags) - 1):
            prev_tag, curr_tag = tags[i], tags[i + 1]
            if prev_tag == 'O':
                if curr_tag == 'O':
                    outside_outside += 1
                else:
                    outside_inside += 1
            else:
                if curr_tag == 'O':
                    inside_outside += 1
                elif prev_tag != curr_tag:
                    cross_type += 1
                else:
                    inside_inside += 1

    transitions = []
    transitions.append(start_outside / (start_outside + start_inside))
    transitions.append(start_inside / (start_outside + start_inside))
    transitions.append(outside_outside / (outside_outside + outside_inside))
    transitions.append(outside_inside / (outside_outside + outside_inside))
    transitions.append(inside_outside / (inside_outside + inside_inside + cross_type))
    transitions.append(inside_inside / (inside_outside + inside_inside + cross_type))
    transitions.append(cross_type / (inside_outside + inside_inside + cross_type))
    return transitions


def linear_warmup(global_step, warmup_steps):
    if global_step < warmup_steps:
        return global_step / warmup_steps
    else:
        return 1.0


class FewShotNERNetwork(nn.Module):
    def __init__(self, sentence_encoder, ignore_label=-1):
        nn.Module.__init__(self)
        self.ignore_label = ignore_label
        self.sentence_encoder = sentence_encoder
        self.loss_function = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, support_set, query_set, class_num, support_num, query_num):
        raise NotImplementedError

    def compute_loss(self, logits, labels):
        class_count = logits.size(-1)
        return self.loss_function(logits.view(-1, class_count), labels.view(-1))

    def __remove_ignore_labels(self, predictions, labels):
        predictions = predictions[labels != self.ignore_label]
        labels = labels[labels != self.ignore_label]
        assert predictions.shape[0] == labels.shape[0]
        return predictions, labels

    def compute_accuracy(self, predictions, labels):
        predictions, labels = self.__remove_ignore_labels(predictions, labels)
        return torch.mean((predictions.view(-1) == labels.view(-1)).type(torch.FloatTensor))

    def __get_class_span_mapping(self, labels, is_string=False):
        class_spans = {}
        current_label = None
        index = 0
        if not is_string:
            while index < len(labels):
                if labels[index] > 0:
                    start_pos = index
                    current_label = labels[index]
                    index += 1
                    while index < len(labels) and labels[index] == current_label:
                        index += 1
                    if current_label in class_spans:
                        class_spans[current_label].append((start_pos, index))
                    else:
                        class_spans[current_label] = [(start_pos, index)]
                else:
                    assert labels[index] == 0
                    index += 1
        else:
            while index < len(labels):
                if labels[index] != 'O':
                    start_pos = index
                    current_label = labels[index]
                    index += 1
                    while index < len(labels) and labels[index] == current_label:
                        index += 1
                    if current_label in class_spans:
                        class_spans[current_label].append((start_pos, index))
                    else:
                        class_spans[current_label] = [(start_pos, index)]
                else:
                    index += 1
        return class_spans

    def __get_entity_intersection(self, pred_spans, label_spans):
        count = 0
        for label in label_spans:
            count += len(list(set(label_spans[label]).intersection(set(pred_spans.get(label, [])))))
        return count

    def __get_entity_count(self, class_spans):
        count = 0
        for label in class_spans:
            count += len(class_spans[label])
        return count

    def __convert_labels_to_tags(self, predictions, query_set):
        pred_tags = []
        label_tags = []
        current_sent_idx = 0
        current_token_idx = 0
        assert len(query_set['sentence_num']) == len(query_set['label2tag'])
        for idx, num in enumerate(query_set['sentence_num']):
            true_labels = torch.cat(query_set['label'][current_sent_idx:current_sent_idx + num], 0)
            true_labels = true_labels[true_labels != self.ignore_label]

            true_labels = true_labels.cpu().numpy().tolist()
            token_count = len(true_labels)
            try:
                pred_tags += [query_set['label2tag'][idx].get(label, 'O') for label in
                              predictions[current_token_idx:current_token_idx + token_count]]
                label_tags += [query_set['label2tag'][idx][label] for label in true_labels]
            except:
                print(predictions[current_token_idx:current_token_idx + token_count], flush=True)
                print(query_set['label2tag'][idx], flush=True)
                exit(0)
            current_sent_idx += num
            current_token_idx += token_count
        assert len(pred_tags) == len(label_tags)
        assert len(pred_tags) == len(predictions)
        return pred_tags, label_tags

    def __get_correct_entity_spans(self, pred_spans, label_spans):
        pred_span_list = []
        label_span_list = []
        for pred in pred_spans:
            pred_span_list += pred_spans[pred]
        for label in label_spans:
            label_span_list += label_spans[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    def __get_coarse_correct_fine_error(self, pred_spans, label_spans):
        count = 0
        for label in label_spans:
            coarse_type = label.split('-')[0]
            relevant_pred_spans = []
            for pred in pred_spans:
                if pred != label and pred.split('-')[0] == coarse_type:
                    relevant_pred_spans += pred_spans[pred]
            count += len(list(set(label_spans[label]).intersection(set(relevant_pred_spans))))
        return count

    def __get_coarse_error_spans(self, pred_spans, label_spans):
        count = 0
        for label in label_spans:
            coarse_type = label.split('-')[0]
            irrelevant_pred_spans = []
            for pred in pred_spans:
                if pred != label and pred.split('-')[0] != coarse_type:
                    irrelevant_pred_spans += pred_spans[pred]
            count += len(list(set(label_spans[label]).intersection(set(irrelevant_pred_spans))))
        return count

    def __get_type_mismatches(self, predictions, labels, query_set):
        pred_tags, label_tags = self.__convert_labels_to_tags(predictions, query_set)
        pred_spans = self.__get_class_span_mapping(pred_tags, is_string=True)
        label_spans = self.__get_class_span_mapping(label_tags, is_string=True)
        total_correct = self.__get_correct_entity_spans(pred_spans, label_spans) + 1e-6
        fine_errors = self.__get_coarse_correct_fine_error(pred_spans, label_spans)
        coarse_errors = self.__get_coarse_error_spans(pred_spans, label_spans)
        return fine_errors, coarse_errors, total_correct

    def entity_level_metrics(self, predictions, labels):
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        predictions, labels = self.__remove_ignore_labels(predictions, labels)
        predictions = predictions.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        pred_spans = self.__get_class_span_mapping(predictions)
        label_spans = self.__get_class_span_mapping(labels)
        pred_count = self.__get_entity_count(pred_spans)
        label_count = self.__get_entity_count(label_spans)
        correct_count = self.__get_entity_intersection(pred_spans, label_spans)
        return pred_count, label_count, correct_count

    def error_analysis_metrics(self, predictions, labels, query_set):
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        predictions, labels = self.__remove_ignore_labels(predictions, labels)
        false_positives = torch.sum(((predictions > 0) & (labels == 0)).type(torch.FloatTensor))
        false_negatives = torch.sum(((predictions == 0) & (labels > 0)).type(torch.FloatTensor))
        predictions = predictions.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        fine_err, coarse_err, total_spans = self.__get_type_mismatches(predictions, labels, query_set)
        return false_positives, false_negatives, len(predictions), fine_err, coarse_err, total_spans

    def set_test_mode(self):
        pass

    def set_validation_mode(self):
        pass

    def set_training_mode(self):
        pass


class FewShotNERManager:

    def __init__(self, training_loader, validation_loader, testing_loader, use_viterbi=False, class_count=None,
                 training_filename=None, temperature=0.05, use_sampled_data=True, contrastive=False, kl_nn_shot=False,
                 extra_loader=None, eval_top_k=None, eval_mix_ratio=None):
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.testing_loader = testing_loader
        self.extra_loader = extra_loader
        self.use_viterbi = use_viterbi
        self.contrastive = contrastive
        self.kl_nn_shot = kl_nn_shot
        self.eval_top_k = eval_top_k
        self.eval_mix_ratio = eval_mix_ratio
        if use_viterbi:
            abstract_trans = calculate_abstract_transitions(training_filename, use_sampled_data=use_sampled_data)
            self.viterbi_solver = ViterbiDecoder(class_count + 2, abstract_trans, temperature)

    def __load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            print("成功加载检查点 '%s'" % checkpoint_path)
            return checkpoint
        else:
            raise Exception("在 '%s' 未找到检查点" % checkpoint_path)

    def extract_scalar(self, x):
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train_model(self,
                    network,
                    network_name,
                    learning_rate=1e-1,
                    training_iterations=30000,
                    validation_iterations=1000,
                    validation_interval=2000,
                    load_checkpoint=None,
                    save_checkpoint=None,
                    warmup_iterations=300,
                    gradient_accumulation=1,
                    fp16_mode=False,
                    use_sgd_for_bert=False,
                    masking_rate=0.2,
                    mask_token_id=103,
                    fine_tuning=False):
        params_to_optimize = list(network.named_parameters())
        no_decay_params = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params_to_optimize = [
            {'params': [p for n, p in params_to_optimize
                        if not any(nd in n for nd in no_decay_params)], 'weight_decay': 0.01},
            {'params': [p for n, p in params_to_optimize
                        if any(nd in n for nd in no_decay_params)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(params_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_iterations,
                                                    num_training_steps=training_iterations)

        if load_checkpoint:
            state_dict = self.__load_checkpoint(load_checkpoint)['state_dict']
            network_state = network.state_dict()
            for name, param in state_dict.items():
                if name not in network_state:
                    print('忽略 {}'.format(name))
                    continue
                print('从 {} 加载 {}'.format(load_checkpoint, name))
                network_state[name].copy_(param)

        if fp16_mode:
            from apex import amp
            network, optimizer = amp.initialize(network, optimizer, opt_level='O1')

        network.train()
        network.set_training_mode()

        best_f1_score = 0.0
        iter_loss_sum = 0.0
        iter_sample_count = 0
        pred_total = 0
        label_total = 0
        correct_total = 0

        current_iter = 0
        while current_iter + 1 < training_iterations:
            for _, (support_batch, query_batch) in enumerate(self.training_loader):
                if torch.cuda.is_available():
                    for key in support_batch:
                        if key in ['word', 'mask', 'text_mask', 'token_mask']:
                            support_batch[key] = support_batch[key].cuda()
                            query_batch[key] = query_batch[key].cuda()

                    def move_to_cuda(tensor_list):
                        return [x.cuda() for x in tensor_list]

                    support_batch['word_tanl'] = move_to_cuda(support_batch['word_tanl'])
                    support_batch['text_mask_tanl'] = move_to_cuda(support_batch['text_mask_tanl'])

                    query_batch['word_tanl'] = move_to_cuda(query_batch['word_tanl'])
                    query_batch['text_mask_tanl'] = move_to_cuda(query_batch['text_mask_tanl'])

                    support_batch['word_aug'] = move_to_cuda(support_batch['word_aug'])
                    support_batch['text_mask_aug'] = move_to_cuda(support_batch['text_mask_aug'])

                    query_batch['word_aug'] = move_to_cuda(query_batch['word_aug'])
                    query_batch['text_mask_aug'] = move_to_cuda(query_batch['text_mask_aug'])

                    labels = torch.cat(query_batch['label'], 0)
                    labels = labels.cuda()
                    device = 'cuda'
                network.cur_it = current_iter
                if masking_rate is not None:
                    for i in range(len(support_batch['word_tanl'])):
                        mask = torch.distributions.Categorical(
                            torch.tensor([1 - masking_rate, masking_rate], device=device)).sample(
                            support_batch['word_tanl'][i].size()).bool()
                        support_batch['word_tanl'][i] = support_batch['word_tanl'][i].masked_fill(
                            mask.masked_fill(support_batch['text_mask_tanl'][i] != -2, False), mask_token_id)

                if self.contrastive:
                    loss_val, logits, predictions = network(support_batch, query_batch, current_iter)
                    loss_val = loss_val / float(gradient_accumulation)
                    assert logits.shape[0] == labels.shape[0], print(
                        logits.shape, labels.shape)
                elif self.kl_nn_shot:
                    kl_logits, logits, predictions = network(support_batch, query_batch)
                    assert logits.shape[0] == labels.shape[0], print(
                        logits.shape, labels.shape)
                    loss_val = network.compute_loss(kl_logits, labels) / float(gradient_accumulation)
                else:
                    logits, predictions = network(support_batch, query_batch)
                    assert logits.shape[0] == labels.shape[0], print(
                        logits.shape, labels.shape)
                    loss_val = network.compute_loss(logits, labels) / float(gradient_accumulation)
                batch_pred_cnt, batch_label_cnt, batch_correct = network.entity_level_metrics(
                    predictions, labels)

                if fp16_mode:
                    with amp.scale_loss(loss_val, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_val.backward()
                if current_iter % gradient_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                iter_loss_sum += loss_val.item()
                pred_total += batch_pred_cnt
                label_total += batch_label_cnt
                correct_total += batch_correct
                iter_sample_count += 1
                if (current_iter + 1) % 100 == 0 or (current_iter + 1) % validation_interval == 0:
                    precision = correct_total / pred_total if correct_total else 0
                    recall = correct_total / label_total if correct_total else 0
                    if precision + recall != 0:
                        f1_score = 2 * precision * recall / (precision + recall)
                    else:
                        f1_score = 0
                    sys.stdout.write('步骤: {0:4} | 损失: {1:2.6f} | [实体] 精确率: {2:3.4f}, 召回率: {3:3.4f}, F1: {4:3.4f}'