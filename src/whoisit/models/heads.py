"""Training heads: they turn embeddings into one value per person.

We only need them for training, later we only use the embedding.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from whoisit.models.backbone import EMBEDDING_SIZE

ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 64.0


class SoftmaxHead(nn.Module):
    def __init__(self, num_people, embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.linear = nn.Linear(embedding_size, num_people)

    def forward(self, embeddings, labels):
        # labels are not used here, but ArcFace needs them, so both heads are called the same way
        return self.linear(embeddings)


class ArcFaceHead(nn.Module):
    def __init__(self, num_people, embedding_size=EMBEDDING_SIZE, margin=ARCFACE_MARGIN,
                 scale=ARCFACE_SCALE):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_people, embedding_size))
        nn.init.xavier_uniform_(self.weights)
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # above pi the cosine would go up again, so we use a straight line from there on
        self.limit = math.cos(math.pi - margin)
        self.drop = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        cosine = F.normalize(embeddings) @ F.normalize(self.weights).T
        # clamp: the square root has no usable gradient at 0
        sine = (1 - cosine ** 2).clamp(min=1e-7).sqrt()
        # cos(theta + m) = cos(theta) cos(m) - sin(theta) sin(m)
        with_margin = cosine * self.cos_m - sine * self.sin_m
        with_margin = torch.where(cosine > self.limit, with_margin, cosine - self.drop)
        # only the correct person gets the margin
        is_correct = F.one_hot(labels, cosine.shape[1]).bool()
        return self.scale * torch.where(is_correct, with_margin, cosine)


# name for --loss -> head class. A new head only has to be added here.
HEADS = {"ce": SoftmaxHead, "arcface": ArcFaceHead}


def make_head(loss, num_people, **options):
    return HEADS[loss](num_people, **options)
