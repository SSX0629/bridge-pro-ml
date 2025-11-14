import torch
import torch.nn as nn


START_ID = 0
O_ID = 1

class ViterbiDecoder:
    def __init__(self, n_tag, abstract_transitions, tau):
        super().__init__()
        self.transitions = self.project_target_transitions(n_tag, abstract_transitions, tau)

    @staticmethod
    def project_target_transitions(n_tag, abstract_transitions, tau):
        s_o, s_i, o_o, o_i, i_o, i_i, x_y = abstract_transitions
        a = torch.eye(n_tag) * i_i
        b = torch.ones(n_tag, n_tag) * x_y / (n_tag - 3)
        c = torch.eye(n_tag) * x_y / (n_tag - 3)
        transitions = a + b - c
        transitions[START_ID, O_ID] = s_o
        transitions[START_ID, O_ID+1:] = s_i / (n_tag - 2)
        transitions[O_ID, O_ID] = o_o
        transitions[O_ID, O_ID+1:] = o_i / (n_tag - 2)
        transitions[O_ID+1:, O_ID] = i_o
        transitions[:, START_ID] = 0.

        powered = torch.pow(transitions, tau)
        summed = powered.sum(dim=1)

        transitions = powered / summed.view(n_tag, 1)

        transitions = torch.where(transitions > 0, transitions, torch.tensor(.000001))

        return torch.log(transitions)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        batch_size, sentence_len, _ = scores.size()

        transitions = self.transitions.expand(batch_size, sentence_len, -1, -1)

        emissions = scores.unsqueeze(2).expand_as(transitions)

        return transitions + emissions

    @staticmethod
    def viterbi(features: torch.Tensor) -> torch.Tensor:
        batch_size, sentence_len, ntags, _ = features.size()

        delta_t = features[:, 0, START_ID, :]
        deltas = [delta_t]

        for t in range(1, sentence_len):
            f_t = features[:, t]
            delta_t, _ = torch.max(f_t + delta_t.unsqueeze(2).expand_as(f_t), 1)
            deltas.append(delta_t)

        sequences = [torch.argmax(deltas[-1], 1, keepdim=True)]
        for t in reversed(range(sentence_len - 1)):
            f_prev = features[:, t + 1].gather(
                2, sequences[-1].unsqueeze(2).expand(batch_size, ntags, 1)).squeeze(2)
            sequences.append(torch.argmax(f_prev + deltas[t], 1, keepdim=True))
        sequences.reverse()
        return torch.cat(sequences, dim=1)