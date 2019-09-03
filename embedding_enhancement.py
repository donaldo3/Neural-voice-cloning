from torch import nn
from hparams import hparams as hp
from speaker_encoder.speaker_encoder3 import FC_ELU

# 2 layer FC-ELU
class EmbeddingEnhancement(nn.Module):
    def __init__(self):
        super(EmbeddingEnhancement, self).__init__()
        self.K = 2
        self.module_list = nn.ModuleList()

        for i in range(self.K):
            self.module_list.append(
                FC_ELU(hp.speaker_embed_dim, hp.speaker_embed_dim)
            )

    def forward(self, x):
        for module in self.module_list:
            x = module(x)

        return x