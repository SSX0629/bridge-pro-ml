import sys

sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F


class PrototypeModel(util.framework.FewShotNERModel):

    def __init__(self, word_encoder, use_dot_product=False, ignore_label=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_label)
        self.dropout = nn.Dropout()
        self.use_dot_product = use_dot_product

    def _calculate_distance(self, vec_a, vec_b, dimension):
        if self.use_dot_product:
            return (vec_a * vec_b).sum(dimension)
        else:
            return -(torch.pow(vec_a - vec_b, 2)).sum(dimension)

    def _batch_distance(self, support_protos, query_embeds, query_mask):
        assert query_embeds.size()[:2] == query_mask.size()
        valid_query_embeds = query_embeds[query_mask == 1].view(-1, query_embeds.size(-1))
        return self._calculate_distance(
            support_protos.unsqueeze(0),
            valid_query_embeds.unsqueeze(1),
            2
        )

    def _get_prototypes(self, embeddings, labels, masks):
        prototypes = []
        valid_embeddings = embeddings[masks == 1].view(-1, embeddings.size(-1))
        concatenated_labels = torch.cat(labels, 0)
        assert concatenated_labels.size(0) == valid_embeddings.size(0)

        for label in range(torch.max(concatenated_labels) + 1):
            class_embeddings = valid_embeddings[concatenated_labels == label]
            prototypes.append(torch.mean(class_embeddings, 0))

        return torch.stack(prototypes)

    def forward(self, support_set, query_set):
        support_embeddings = self.word_encoder(support_set['word'],
                                               support_set['mask'])
        query_embeddings = self.word_encoder(query_set['word'], query_set['mask'])

        support_embeddings = self.dropout(support_embeddings)
        query_embeddings = self.dropout(query_embeddings)

        logit_list = []
        support_pos = 0
        query_pos = 0

        assert support_embeddings.size()[:2] == support_set['mask'].size()
        assert query_embeddings.size()[:2] == query_set['mask'].size()

        for i, support_sent_count in enumerate(support_set['sentence_num']):
            query_sent_count = query_set['sentence_num'][i]

            current_protos = self._get_prototypes(
                support_embeddings[support_pos:support_pos + support_sent_count],
                support_set['label'][support_pos:support_pos + support_sent_count],
                support_set['text_mask'][support_pos:support_pos + support_sent_count]
            )

            logit_list.append(self._batch_distance(
                current_protos,
                query_embeddings[query_pos:query_pos + query_sent_count],
                query_set['text_mask'][query_pos:query_pos + query_sent_count]
            ))

            query_pos += query_sent_count
            support_pos += support_sent_count

        logits = torch.cat(logit_list, 0)
        _, predictions = torch.max(logits, 1)
        return logits, predictions