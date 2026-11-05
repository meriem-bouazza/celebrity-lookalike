"""Training heads: they turn embeddings into one value per person.

We only need them for training, later we only use the embedding.
ArcFaceHead comes later in this file (separate issue).
"""
import torch.nn as nn

from whoisit.models.backbone import EMBEDDING_SIZE


class SoftmaxHead(nn.Module):
    def __init__(self, num_people, embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.linear = nn.Linear(embedding_size, num_people)

    def forward(self, embeddings, labels):
        # labels are not used here, but ArcFace needs them, so both heads are called the same way
        return self.linear(embeddings)


# name for --loss -> head class. A new head only has to be added here.
HEADS = {"ce": SoftmaxHead}


def make_head(loss, num_people):
    return HEADS[loss](num_people)
