import sys

sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F


class FewShotNN(util.framework.FewShotNERModel):
    def __init__(self, token_encoder, use_dot_product=False, skip_index=-1):
        util.framework.FewShotNERModel.__init__(self, token_encoder, ignore_index=skip_index)
        self.dropout = nn.Dropout()
        self.use_dot = use_dot_product

    def _calculate_distance(self, vec_a, vec_b, dimension):
        if self.use_dot:
            return (vec_a * vec_b).sum(dimension)
        else:
            return -(torch.pow(vec_a - vec_b, 2)).sum(dimension)

    def _batch_calculate_distance(self, support_vecs, query_vecs, query_mask):
        assert query_vecs.size()[:2] == query_mask.size()
        query_vecs = query_vecs[query_mask == 1].view(-1, query_vecs.size(-1))
        return self._calculate_distance(support_vecs.unsqueeze(0), query_vecs.unsqueeze(1), 2)

    def _get_closest_distances(self, embeddings, labels, masks, query_embeds, query_masks):
        closest_dists = []
        support_vecs = embeddings[masks == 1].view(-1, embeddings.size(-1))
        labels = torch.cat(labels, 0)
        assert labels.size(0) == support_vecs.size(0)
        dist_matrix = self._batch_calculate_distance(support_vecs, query_embeds, query_masks)
        for label in range(torch.max(labels) + 1):
            closest_dists.append(torch.topk(dist_matrix[:, labels == label], k=1, dim=1).values.mean(dim=1))
        closest_dists = torch.stack(closest_dists, dim=1)
        return closest_dists

    def forward(self, support_set, query_set):
        support_embeddings = self.token_encoder(support_set['word'], support_set['mask'])
        query_embeddings = self.token_encoder(query_set['word'], query_set['mask'])
        support_embeddings = self.dropout(support_embeddings)
        query_embeddings = self.dropout(query_embeddings)

        logit_list = []
        current_support_idx = 0
        current_query_idx = 0
        assert support_embeddings.size()[:2] == support_set['mask'].size()
        assert query_embeddings.size()[:2] == query_set['mask'].size()

        for i, support_sent_count in enumerate(support_set['sentence_num']):
            query_sent_count = query_set['sentence_num'][i]
            logit_list.append(self._get_closest_distances(
                support_embeddings[current_support_idx:current_support_idx + support_sent_count],
                support_set['label'][current_support_idx:current_support_idx + support_sent_count],
                support_set['text_mask'][current_support_idx: current_support_idx + support_sent_count],
                query_embeddings[current_query_idx:current_query_idx + query_sent_count],
                query_set['text_mask'][current_query_idx: current_query_idx + query_sent_count]
            ))
            current_query_idx += query_sent_count
            current_support_idx += support_sent_count
        logit_list = torch.cat(logit_list, 0)
        _, predictions = torch.max(logit_list, 1)
        return logit_list, predictions