import torch
import torch.utils.data as data
import os
from .fewshotsampler import FewshotSampler, FewshotSampleBase
import numpy as np
import json


def extract_category(raw_label):
    if raw_label.startswith('B-') or raw_label.startswith('I-'):
        return raw_label[2:]
    else:
        return raw_label


class DataInstance(FewshotSampleBase):
    def __init__(self, file_lines):
        file_lines = [line.split('\t') for line in file_lines]
        self.tokens, self.labels = zip(*file_lines)
        self.tokens = [token.lower() for token in self.tokens]
        self.standardized_labels = list(map(extract_category, self.labels))
        self.category_count = {}

    def __count_entities__(self):
        current_label = self.standardized_labels[0]
        for label in self.standardized_labels[1:]:
            if label == current_label:
                continue
            else:
                if current_label != 'O':
                    if current_label in self.category_count:
                        self.category_count[current_label] += 1
                    else:
                        self.category_count[current_label] = 1
                current_label = label
        if current_label != 'O':
            if current_label in self.category_count:
                self.category_count[current_label] += 1
            else:
                self.category_count[current_label] = 1

    def get_category_count(self):
        if self.category_count:
            return self.category_count
        else:
            self.__count_entities__()
            return self.category_count

    def get_label_categories(self):
        label_categories = list(set(self.standardized_labels))
        if 'O' in label_categories:
            label_categories.remove('O')
        return label_categories

    def is_valid(self, target_categories):
        return (set(self.get_category_count().keys()).intersection(set(target_categories))) and not (
            set(self.get_category_count().keys()).difference(set(target_categories)))

    def __str__(self):
        new_lines = zip(self.tokens, self.labels)
        return '\n'.join(['\t'.join(line) for line in new_lines])


class FewShotNERDatasetWithRandomSelection(data.Dataset):
    """
    Fewshot NER Dataset
    """

    def __init__(self, data_path, tokenizer, num_categories, num_support, num_query, max_seq_len, ignore_label_val=-1,
                 i2b2_flag=False, dataset_title=None, no_random=False, no_separator=False):
        if not os.path.exists(data_path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.category_to_instance_id = {}
        self.num_categories = num_categories
        self.num_support = num_support
        self.num_query = num_query
        self.tokenizer = tokenizer
        self.instances, self.categories = self.__load_data_from_path__(data_path)
        self.max_seq_len = max_seq_len
        self.selector = FewshotSampler(num_categories, num_support, num_query, self.instances, classes=self.categories,
                                       i2b2flag=i2b2_flag, dataset_name=dataset_title, no_shuffle=no_random)
        self.ignore_label_val = ignore_label_val
        self.no_separator = no_separator

        print(data_path, len(self.categories), self.categories, flush=True)

    def __insert_instance__(self, index, instance_categories):
        for item in instance_categories:
            if item in self.category_to_instance_id:
                self.category_to_instance_id[item].append(index)
            else:
                self.category_to_instance_id[item] = [index]

    def __load_data_from_path__(self, data_path):
        instances = []
        categories = []
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        instance_lines = []
        index = 0
        for line in lines:
            line = line.strip()
            if line:
                instance_lines.append(line)
            else:
                instance = DataInstance(instance_lines)
                instances.append(instance)
                instance_categories = instance.get_label_categories()
                self.__insert_instance__(index, instance_categories)
                categories += instance_categories
                instance_lines = []
                index += 1

        if len(instance_lines) != 0:
            instance = DataInstance(instance_lines)
            instances.append(instance)
            instance_categories = instance.get_label_categories()
            self.__insert_instance__(index, instance_categories)
            categories += instance_categories
            instance_lines = []
            index += 1
        categories = list(set(categories))
        return instances, categories

    def __get_token_label_pairs__(self, instance):
        tokens = []
        labels = []
        for token, label in zip(instance.tokens, instance.standardized_labels):
            token_pieces = self.tokenizer.tokenize(token)
            if token_pieces:
                tokens.extend(token_pieces)
                token_labels = [self.label_to_id[label]] + [self.ignore_label_val] * (len(token_pieces) - 1)
                labels.extend(token_labels)
        return tokens, labels

    def build_prompt_tanl(self, tokens, labels):
        tanl_tokens = []
        tanl_text_mask = []
        tanl_labels = []
        last_tag = None

        for token, label in zip(tokens, labels):
            if last_tag is not None and last_tag != label and last_tag > 0:
                label_sequence = self.id_to_label[last_tag].replace('-', ' - ')
                label_sequence = label_sequence.replace('/', ' / ')
                if self.no_separator:
                    tanl_tokens.extend(label_sequence.split())
                    tanl_text_mask.extend([-2] * len(label_sequence.split()))
                    tanl_labels.extend([label] * len(label_sequence.split()))
                else:
                    tanl_tokens.extend(['|'] + label_sequence.split() + [']'])
                    tanl_text_mask.extend([0] + [-2] * len(label_sequence.split()) + [0])
                    tanl_labels.extend([label] * (len(label_sequence.split()) + 2))
            if label > 0 and (last_tag is None or last_tag != label):
                if not self.no_separator:
                    tanl_tokens.extend(['['])
                    tanl_text_mask.extend([0])
                    tanl_labels.extend([label])

            tanl_tokens.extend([token])
            tanl_text_mask.extend([1])
            tanl_labels.extend([label])
            last_tag = label

        if last_tag is not None and last_tag > 0:
            label_sequence = self.id_to_label[last_tag].replace('-', ' - ')
            label_sequence = label_sequence.replace('/', ' / ')
            if self.no_separator:
                tanl_tokens.extend(label_sequence.split())
                tanl_text_mask.extend([-2] * len(label_sequence.split()))
                tanl_labels.extend([label] * len(label_sequence.split()))
            else:
                tanl_tokens.extend(['|'] + label_sequence.split() + [']'])
                tanl_text_mask.extend([0] + [-2] * len(label_sequence.split()) + [0])
                tanl_labels.extend([last_tag] * (len(label_sequence.split()) + 2))
        return tanl_tokens, tanl_text_mask, tanl_labels

    def build_prompt_augmented(self, tokens, labels):
        entities = [self.id_to_label[opt].replace('-', ' - ').replace('/', ' / ') for opt in
                    range(torch.max(torch.tensor(labels)) + 1)]

        if self.no_separator:
            aug_tokens = ' '.join(entities).split()
            aug_text_mask = [0] * len(aug_tokens)
            aug_labels = labels
        else:
            aug_tokens = ' , '.join(entities).split() + [':']
            aug_text_mask = [0] * len(aug_tokens)
            aug_labels = labels

        aug_tokens.extend(tokens)
        aug_text_mask.extend([1] * len(tokens))

        return aug_tokens, aug_text_mask, aug_labels

    def __get_raw_data__(self, tokens, labels):
        tokens_list = []
        labels_list = []
        while len(tokens) > self.max_seq_len - 2:
            tokens_list.append(tokens[:self.max_seq_len - 2])
            tokens = tokens[self.max_seq_len - 2:]
            labels_list.append(labels[:self.max_seq_len - 2])
            labels = labels[self.max_seq_len - 2:]
        if tokens:
            tokens_list.append(tokens)
            labels_list.append(labels)

        indexed_tokens_list = []
        mask_list = []
        text_mask_list = []
        token_mask_list = []

        indexed_tanl_tokens_list = []
        tanl_text_mask_list = []
        tanl_label_list = []

        indexed_aug_tokens_list = []
        aug_text_mask_list = []
        aug_label_list = []
        for i, tokens in enumerate(tokens_list):
            tanl_tokens, tanl_text_mask, tanl_labels = self.build_prompt_tanl(tokens, labels_list[i])
            aug_tokens, aug_text_mask, aug_labels = self.build_prompt_augmented(tokens, labels_list[i])

            tanl_tokens = ['[CLS]'] + tanl_tokens + ['[SEP]']
            aug_tokens = ['CLS'] + aug_tokens + ['[SEP]']

            tokens = ['[CLS]'] + tokens + ['[SEP]']

            tanl_text_mask = [0] + tanl_text_mask + [0]
            aug_text_mask = [0] + aug_text_mask + [0]

            indexed_tanl_tokens_list.append(self.tokenizer.convert_tokens_to_ids(tanl_tokens))
            indexed_aug_tokens_list.append(self.tokenizer.convert_tokens_to_ids(aug_tokens))
            tanl_text_mask_list.append(tanl_text_mask)
            aug_text_mask_list.append(aug_text_mask)

            tanl_label_list.append(tanl_labels)
            aug_label_list.append(aug_labels)

            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            while len(indexed_tokens) < self.max_seq_len:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            mask = np.zeros((self.max_seq_len), dtype=np.int32)
            mask[:len(tokens)] = 1
            mask_list.append(mask)

            text_mask = np.zeros((self.max_seq_len), dtype=np.int32)
            text_mask[1:len(tokens) - 1] = 1

            token_mask = np.zeros((self.max_seq_len), dtype=np.int32)
            token_mask[1:len(tokens) - 1] = 1

            for j, t in enumerate(labels_list[i]):
                if t != 0:
                    token_mask[1 + j] = 0
            text_mask_list.append(text_mask)
            token_mask_list.append(token_mask)
            assert len(labels_list[i]) == len(tokens) - 2, print(labels_list[i], tokens)
        return indexed_tokens_list, mask_list, text_mask_list, labels_list, token_mask_list, indexed_tanl_tokens_list, tanl_text_mask_list, tanl_label_list, indexed_aug_tokens_list, aug_text_mask_list, aug_label_list

    def __add_entry__(self, index, data_dict, word_data, mask_data, text_mask_data, label_data, token_mask_data,
                      tanl_word_data, tanl_text_mask_data, tanl_label_data, aug_word_data, aug_text_mask_data,
                      aug_label_data):
        data_dict['index'].append(index)
        data_dict['word'] += word_data
        data_dict['mask'] += mask_data
        data_dict['label'] += label_data
        data_dict['text_mask'] += text_mask_data
        data_dict['token_mask'] += token_mask_data
        data_dict['word_tanl'] += tanl_word_data
        data_dict['text_mask_tanl'] += tanl_text_mask_data
        data_dict['label_tanl'] += tanl_label_data
        data_dict['word_aug'] += aug_word_data
        data_dict['text_mask_aug'] += aug_text_mask_data
        data_dict['label_aug'] += aug_label_data

    def __populate_dataset__(self, idx_list, save_label_map=False):
        dataset = {'index': [], 'word': [], 'mask': [], 'label': [], 'sentence_num': [], 'text_mask': [],
                   'token_mask': [], 'word_tanl': [], 'text_mask_tanl': [], 'label_tanl': [], 'word_aug': [],
                   'text_mask_aug': [], 'label_aug': []}
        for idx in idx_list:
            tokens, labels = self.__get_token_label_pairs__(self.instances[idx])
            word_data, mask_data, text_mask_data, label_data, token_mask_data, tanl_word_data, tanl_text_mask_data, tanl_label_data, aug_word_data, aug_text_mask_data, aug_label_data = self.__get_raw_data__(
                tokens, labels)
            word_data = torch.tensor(word_data).long()
            mask_data = torch.tensor(mask_data).long()

            text_mask_data = torch.tensor(text_mask_data).long()
            token_mask_data = torch.tensor(token_mask_data).long()

            tanl_word_data = [torch.tensor(x).long() for x in tanl_word_data]
            tanl_text_mask_data = [torch.tensor(x).long() for x in tanl_text_mask_data]

            aug_word_data = [torch.tensor(x).long() for x in aug_word_data]
            aug_text_mask_data = [torch.tensor(x).long() for x in aug_text_mask_data]

            self.__add_entry__(idx, dataset, word_data, mask_data, text_mask_data, label_data, token_mask_data,
                               tanl_word_data, tanl_text_mask_data, tanl_label_data, aug_word_data, aug_text_mask_data,
                               aug_label_data)
        dataset['sentence_num'] = [len(dataset['word'])]
        if save_label_map:
            dataset['label2tag'] = [self.id_to_label]
            dataset['tag2label'] = [self.label_to_id]
        return dataset

    def __getitem__(self, index):
        target_categories, support_idx, query_idx = self.selector.__next__()
        distinct_labels = ['O'] + target_categories
        self.label_to_id = {label: idx for idx, label in enumerate(distinct_labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(distinct_labels)}

        support_set = self.__populate_dataset__(support_idx)
        query_set = self.__populate_dataset__(query_idx, save_label_map=True)
        return support_set, query_set

    def __len__(self):
        return 100000


class SimpleSelector(FewShotNERDatasetWithRandomSelection):
    def __init__(self, data_path, tokenizer, num_categories, num_support, num_query, max_seq_len, ignore_label_val=-1):
        super(SimpleSelector, self).__init__(data_path, tokenizer, num_categories, num_support, num_query, max_seq_len,
                                             ignore_label_val=-1)

    def set_label_mapping(self, label_to_id=None, id_to_label=None):
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label

    def __getitem__(self, index):
        query_idx = list(range(index * 16, (index + 1) * 16))
        target_categories = list(set([x for idx in query_idx for x in self.instances[idx].get_category_count().keys()]))
        distinct_labels = ['O'] + target_categories
        query_set = self.__populate_dataset__(query_idx, save_label_map=True)
        return query_set

    def __len__(self):
        print('hello', len(self.instances))
        return len(self.instances) // 16


class SimpleSelectorForSupport(FewShotNERDatasetWithRandomSelection):
    def __init__(self, data_path, tokenizer, num_categories, num_support, num_query, max_seq_len, ignore_label_val=-1):
        super(SimpleSelectorForSupport, self).__init__(data_path, tokenizer, num_categories, num_support, num_query,
                                                       max_seq_len, ignore_label_val=-1)

    def __getitem__(self, index):
        support_idx = list(range(len(self.instances)))
        target_categories = list(
            set([x for idx in support_idx for x in self.instances[idx].get_category_count().keys()]))
        distinct_labels = ['O'] + target_categories
        print(distinct_labels, flush=True)
        self.label_to_id = {label: idx for idx, label in enumerate(distinct_labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(distinct_labels)}

        support_set = query_set = self.__populate_dataset__(support_idx, save_label_map=True)
        return support_set, query_set

    def __len__(self):
        return 1


class FewShotNERDataset(FewShotNERDatasetWithRandomSelection):
    def __init__(self, data_path, tokenizer, max_seq_len, ignore_label_val=-1, no_random=False, no_separator=False):
        if not os.path.exists(data_path):
            assert (0)
        self.category_to_instance_id = {}
        self.tokenizer = tokenizer
        self.instances, self.categories = self.__load_data_from_path__(data_path)
        self.max_seq_len = max_seq_len
        self.ignore_label_val = ignore_label_val
        self.no_random = no_random
        self.no_separator = no_separator

    def __load_data_from_path__(self, data_path):
        with open(data_path) as f:
            lines = f.readlines()
        categories = []
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i].strip())
            categories += lines[i]['types']

        categories = list(set(categories))
        return lines, categories

    def __add_entry__(self, index, data_dict, word_data, mask_data, text_mask_data, label_data, token_mask_data,
                      tanl_word_data, tanl_text_mask_data, tanl_label_data, aug_word_data, aug_text_mask_data,
                      aug_label_data):
        data_dict['index'].append(index)
        data_dict['word'] += word_data
        data_dict['mask'] += mask_data
        data_dict['label'] += label_data
        data_dict['text_mask'] += text_mask_data
        data_dict['token_mask'] += token_mask_data
        data_dict['word_tanl'] += tanl_word_data
        data_dict['text_mask_tanl'] += tanl_text_mask_data
        data_dict['label_tanl'] += tanl_label_data
        data_dict['word_aug'] += aug_word_data
        data_dict['text_mask_aug'] += aug_text_mask_data
        data_dict['label_aug'] += aug_label_data

    def __get_token_label_pairs__(self, tokens, labels):
        token_list = []
        label_list = []
        for token, label in zip(tokens, labels):
            token_pieces = self.tokenizer.tokenize(token)
            if token_pieces:
                token_list.extend(token_pieces)
                token_labels = [self.label_to_id[label]] + [self.ignore_label_val] * (len(token_pieces) - 1)
                label_list.extend(token_labels)
        return token_list, label_list

    def __populate_dataset__(self, data, save_label_map=False):
        dataset = {'index': [], 'word': [], 'mask': [], 'label': [], 'sentence_num': [], 'text_mask': [],
                   'token_mask': [], 'word_tanl': [], 'text_mask_tanl': [], 'label_tanl': [], 'word_aug': [],
                   'text_mask_aug': [], 'label_aug': []}
        for i in range(len(data['word'])):
            tokens, labels = self.__get_token_label_pairs__(data['word'][i], data['label'][i])
            word_data, mask_data, text_mask_data, label_data, token_mask_data, tanl_word_data, tanl_text_mask_data, tanl_label_data, aug_word_data, aug_text_mask_data, aug_label_data = self.__get_raw_data__(
                tokens, labels)
            word_data = torch.tensor(word_data).long()
            mask_data = torch.tensor(mask_data).long()
            text_mask_data = torch.tensor(text_mask_data).long()
            token_mask_data = torch.tensor(token_mask_data).long()

            tanl_word_data = [torch.tensor(x).long() for x in tanl_word_data]
            tanl_text_mask_data = [torch.tensor(x).long() for x in tanl_text_mask_data]

            aug_word_data = [torch.tensor(x).long() for x in aug_word_data]
            aug_text_mask_data = [torch.tensor(x).long() for x in aug_text_mask_data]

            self.__add_entry__(i, dataset, word_data, mask_data, text_mask_data, label_data, token_mask_data,
                               tanl_word_data, tanl_text_mask_data, tanl_label_data, aug_word_data, aug_text_mask_data,
                               aug_label_data)
        dataset['sentence_num'] = [len(dataset['word'])]
        if save_label_map:
            dataset['label2tag'] = [self.id_to_label]
            dataset['tag2label'] = [self.label_to_id]
        return dataset

    def __getitem__(self, index):
        instance = self.instances[index]
        target_categories = self.categories if self.no_random else instance['types']
        support = instance['support']
        query = instance['query']
        distinct_labels = ['O'] + target_categories
        self.label_to_id = {label: idx for idx, label in enumerate(distinct_labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(distinct_labels)}
        support_set = self.__populate_dataset__(support)
        query_set = self.__populate_dataset__(query, save_label_map=True)
        return support_set, query_set

    def __len__(self):
        return len(self.instances)


def collate_function(data):
    batch_support = {'index': [], 'word': [], 'mask': [], 'label': [], 'sentence_num': [], 'text_mask': [],
                     'token_mask': [], 'word_tanl': [], 'text_mask_tanl': [], 'label_tanl': [], 'word_aug': [],
                     'text_mask_aug': [], 'label_aug': []}
    batch_query = {'index': [], 'word': [], 'mask': [], 'label': [], 'sentence_num': [], 'label2tag': [],
                   'tag2label': [], 'text_mask': [], 'token_mask': [], 'word_tanl': [], 'text_mask_tanl': [],
                   'label_tanl': [], 'word_aug': [], 'text_mask_aug': [], 'label_aug': []}
    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in batch_support:
            batch_support[k] += support_sets[i][k]
        for k in batch_query:
            batch_query[k] += query_sets[i][k]

    for k in batch_support:
        if k in ['word', 'mask', 'text_mask', 'token_mask']:
            batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        if k in ['word', 'mask', 'text_mask', 'token_mask']:
            batch_query[k] = torch.stack(batch_query[k], 0)

    batch_support['label'] = [torch.tensor(tag_list).long() for tag_list in batch_support['label']]
    batch_query['label'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label']]

    batch_support['label_tanl'] = [torch.tensor(tag_list).long() for tag_list in batch_support['label_tanl']]
    batch_query['label_tanl'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label_tanl']]

    batch_support['label_aug'] = [torch.tensor(tag_list).long() for tag_list in batch_support['label_aug']]