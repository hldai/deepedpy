import torch
from torch import nn
import torch.nn.functional as F


class DescHypEmbed(nn.Module):
    def __init__(self, word_vecs, n_entities, init_enity_vecs):
        super(DescHypEmbed, self).__init__()

        self.word_embed_layer = nn.Embedding.from_pretrained(torch.from_numpy(word_vecs))
        self.word_embed_layer.weight.requires_grad = False

        self.dim = self.word_embed_layer.embedding_dim
        # self.entity_embed_layer = nn.Embedding(n_entities, self.dim)
        self.entity_embed_layer = nn.Embedding.from_pretrained(torch.from_numpy(init_enity_vecs))
        self.entity_embed_layer.weight.requires_grad = True

    def forward(self, batch_size, word_ids, entity_ids):
        word_vecs = self.word_embed_layer(word_ids)
        word_vecs = F.normalize(word_vecs, dim=2)
        entity_vecs = F.normalize(self.entity_embed_layer(entity_ids), dim=1)
        return torch.bmm(word_vecs, entity_vecs.view(-1, self.dim, 1))
