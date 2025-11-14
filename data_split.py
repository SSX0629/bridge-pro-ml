# %%
import random
from util.data_loader import Sample, get_class_name
from sklearn.model_selection import train_test_split
import os
import numpy as np

np.random.seed(0)
random.seed(0)

sample_modes = ['5-shot', '10%', '100%']
split_modes = ['supervised', 'intra', 'inter']


class ConllSample(Sample):
    def __init__(self, raw_lines):
        self.line_list = [[line.split(' ')[0], line.split(' ')[-1]] for line in raw_lines]
        self.word_list, self.tag_list = zip(*self.line_list)


class MySample(Sample):
    def get_coarse_tag_classes(self):
        return list(set([tag.split('-')[0] for tag in self.tag_list if tag != 'O']))

    def __search_class__(self, tag, target_categories):
        for category_name in target_categories:
            if tag.startswith(category_name):
                return category_name
        return None

    def reassign_coarse_tags(self, target_categories):
        new_tag_list = []
        for tag in self.normalized_tags:
            category_name = self.__search_class__(tag, target_categories)
            if category_name:
                new_tag_list.append(tag)
            else:
                new_tag_list.append('O')
        self.updated_tags = new_tag_list

    def reassign_tags(self, target_categories):
        new_tag_list = []
        for tag in self.normalized_tags:
            if tag in target_categories:
                new_tag_list.append(tag)
            else:
                new_tag_list.append('O')
        self.updated_tags = new_tag_list


class DataSplitter:
    def __init__(self, input_path):
        self.category_to_sample_ids = {}
        self.coarse_category_to_sample_ids = {}
        self.sample_list = []
        self.input_path = input_path

    def __add_sample_to_dict__(self, dict_obj, index, sample_categories):
        for item in sample_categories:
            if item in dict_obj:
                dict_obj[item].append(index)
            else:
                dict_obj[item] = [index]

    def __save_to_file__(self, sample_collection, file_path):
        print(file_path, len(sample_collection))
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n\n'.join([str(sample) for sample in sample_collection]))

    def __get_sample_indices__(self, dict_obj, category_collection):
        indices = []
        for category_name in category_collection:
            indices += dict_obj[category_name]
        return indices

    def __update_tags__(self, sample_collection, target_categories, split_type):
        if split_type == 'intra':
            for sample in sample_collection:
                sample.reassign_coarse_tags(target_categories)
        else:
            for sample in sample_collection:
                sample.reassign_tags(target_categories)

    def load_conll_data(self):
        with open(self.input_path, 'r') as f:
            lines = f.readlines()
        current_sample_lines = []
        index = 0
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('-DOCSTART-'):
                    continue
                current_sample_lines.append(line)
            else:
                if current_sample_lines:
                    sample = ConllSample(current_sample_lines)
                    self.sample_list.append(sample)
                    sample_categories = sample.get_tag_class()
                    self.__add_sample_to_dict__(self.category_to_sample_ids, index, sample_categories)
                    index += 1
                current_sample_lines = []

    def load_custom_data(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        current_sample_lines = []
        index = 0
        for line in lines:
            line = line.strip()
            if line:
                current_sample_lines.append(line)
            else:
                if current_sample_lines:
                    sample = MySample(current_sample_lines)
                    self.sample_list.append(sample)
                    sample_categories = sample.get_tag_class()
                    coarse_sample_categories = sample.get_coarse_tag_classes()
                    self.__add_sample_to_dict__(self.category_to_sample_ids, index, sample_categories)
                    self.__add_sample_to_dict__(self.coarse_category_to_sample_ids, index, coarse_sample_categories)
                    index += 1
                current_sample_lines = []

    def split_dataset(self, split_type, params):
        if split_type not in split_modes:
            return
        if split_type == 'supervised':
            train_val, test_set, _, _ = train_test_split(self.sample_list, [0] * len(self.sample_list), test_size=0.2,
                                                         random_state=0)
            train_set, val_set, _, _ = train_test_split(train_val, [0] * len(train_val), test_size=0.125,
                                                        random_state=0)
        else:
            if split_type == 'intra':
                train_indices = self.__get_sample_indices__(self.coarse_category_to_sample_ids,
                                                            params['train-categories'])
                val_indices = self.__get_sample_indices__(self.coarse_category_to_sample_ids, params['val-categories'])
                test_indices = self.__get_sample_indices__(self.coarse_category_to_sample_ids,
                                                           params['test-categories'])
            else:
                if not params['train-categories']:
                    train_categories = []
                    val_categories = []
                    test_categories = []
                    for coarse_category in self.coarse_category_to_sample_ids:
                        fine_categories = [fine_category for fine_category in self.category_to_sample_ids if
                                           fine_category.startswith(coarse_category)]
                        length = len(fine_categories)
                        if length < 3:
                            print('在[{}]中没有足够的细粒度类别，验证集或测试集可能不包含粗类别[{}]'.format(
                                coarse_category, coarse_category))
                        permuted = np.random.permutation(length)
                        train_categories += list(fine_categories[i] for i in permuted[:(max(int(length * 0.6), 1))])
                        val_categories += list(
                            fine_categories[i] for i in permuted[max(int(length * 0.6), 1):max(int(length * 0.8), 2)])
                        test_categories += list(fine_categories[i] for i in permuted[max(int(length * 0.8), 2):])

                    params['train-categories'] = train_categories
                    params['val-categories'] = val_categories
                    params['test-categories'] = test_categories
                    print(train_categories)
                    print(val_categories)
                    print(test_categories)
                train_indices = self.__get_sample_indices__(self.category_to_sample_ids, params['train-categories'])
                val_indices = self.__get_sample_indices__(self.category_to_sample_ids, params['val-categories'])
                test_indices = self.__get_sample_indices__(self.category_to_sample_ids, params['test-categories'])
            train_indices = list(set(train_indices))
            val_indices = list(set(val_indices).difference(set(train_indices)))
            test_indices = list(set(test_indices).difference(set(train_indices + val_indices)))
            train_set = [self.sample_list[i] for i in train_indices]
            val_set = [self.sample_list[i] for i in val_indices]
            test_set = [self.sample_list[i] for i in test_indices]
            self.__update_tags__(train_set, params['train-categories'], split_type)
            self.__update_tags__(val_set, params['val-categories'], split_type)
            self.__update_tags__(test_set, params['test-categories'], split_type)
        self.__save_to_file__(train_set, params['train-path'])
        self.__save_to_file__(val_set, params['val-path'])
        self.__save_to_file__(test_set, params['test-path'])

    def process_samples(self, sample_type, output_path):
        if sample_type not in sample_modes:
            print('错误的采样模式 {}'.format(sample_type))
            return
        if sample_type == '5-shot':
            selected_ids = []
            for category_name in self.category_to_sample_ids:
                selected_ids += random.sample(self.category_to_sample_ids[category_name], 5)
            selected_ids = list(set(selected_ids))
            chosen_samples = [self.sample_list[i] for i in selected_ids]
        elif sample_type == '10%':
            chosen_samples = random.sample(self.sample_list, int(len(self.sample_list) * 0.1))
        else:
            chosen_samples = self.sample_list
        print(len(chosen_samples))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines('\n\n'.join([str(sample) for sample in chosen_samples]))


# %%
if __name__ == '__main__':
    params = {'train-path': '', 'val-path': '', 'test-path': '', 'train-categories': [], 'val-categories': [],
              'test-categories': []}
    params['train-path'] = 'data/mydata/train-intra-new.txt'
    params['val-path'] = 'data/mydata/val-intra-new.txt'
    params['test-path'] = 'data/mydata/test-intra-new.txt'
    params['train-categories'] = ['person', 'other', 'art', 'product']
    params['val-categories'] = ['event', 'building']
    params['test-categories'] = ['organization', 'location']

    if not os.path.exists('data/mydata/'):
        os.mkdir('data/mydata/')
    splitter = DataSplitter('data/processed_data_0131')
    splitter.load_custom_data()
    splitter.split_dataset('intra', params)

# %%