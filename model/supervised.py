import sys

sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F


class TransferBERT(util.framework.FewShotNERModel):

    def __init__(self, word_encoder, use_dot_product=False, ignore_label=-1, training_classes=None,
                 validation_classes=None, testing_classes=None):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_label)
        self.dropout = nn.Dropout()
        self.use_dot_product = use_dot_product
        self.training_head = nn.Linear(768, training_classes + 1)
        self.validation_head = nn.Linear(768, validation_classes + 1)
        self.testing_head = nn.Linear(768, testing_classes + 1)
        self.classifier_head = self.training_head
        self.state_flag = 0
        self.training_classes = training_classes
        self.validation_classes = validation_classes
        self.testing_classes = testing_classes

    def _calculate_distance(self, vec_a, vec_b, dimension):
        if self.use_dot_product:
            return (vec_a * vec_b).sum(dimension)
        else:
            return -(torch.pow(vec_a - vec_b, 2)).sum(dimension)

    def _batch_distance(self, support_vectors, query_vectors, query_mask):
        assert query_vectors.size()[:2] == query_mask.size()
        valid_query_vectors = query_vectors[query_mask == 1].view(-1, query_vectors.size(-1))
        return self._calculate_distance(support_vectors.unsqueeze(0), valid_query_vectors.unsqueeze(1), 2)

    def _get_closest_distances(self, embeddings, labels, mask, query_vectors, query_mask):
        closest_distances = []
        valid_support_vectors = embeddings[mask == 1].view(-1, embeddings.size(-1))
        concatenated_labels = torch.cat(labels, 0)
        assert concatenated_labels.size(0) == valid_support_vectors.size(0)
        distance_matrix = self._batch_distance(valid_support_vectors, query_vectors, query_mask)

        for label in range(torch.max(concatenated_labels) + 1):
            label_distances = distance_matrix[:, concatenated_labels == label]
            top1_distances = torch.topk(label_distances, k=1, dim=1).values
            closest_distances.append(top1_distances.mean(dim=1))

        closest_distances = torch.stack(closest_distances, dim=1)
        return closest_distances

    def activate_validation_mode(self):
        self.classifier_head = self.validation_head

    def activate_test_mode(self):
        self.classifier_head = self.testing_head

    def activate_training_mode(self):
        self.classifier_head = self.training_head

    def forward(self, support_set, query_set):
        query_embeddings = self.word_encoder(query_set['word'], query_set['mask'])  # [num_sent, number_of_tokens, 768]
        query_embeddings = self.dropout(query_embeddings)

        logit_list = []

        current_query_count = 0
        assert query_embeddings.size()[:2] == query_set['mask'].size()

        for i, support_sent_count in enumerate(support_set['sentence_num']):
            query_sent_count = query_set['sentence_num'][i]
            current_queries = query_embeddings[current_query_count: current_query_count + query_sent_count]
            current_query_masks = query_set['text_mask'][current_query_count: current_query_count + query_sent_count]

            valid_query_embeds = current_queries[current_query_masks == 1].view(-1, current_queries.size(-1))
            logit_list.append(self.classifier_head(valid_query_embeds))

            current_query_count += query_sent_count

        logits = torch.cat(logit_list, dim=0)
        _, predictions = torch.max(logits, 1)
        return logits, predictions