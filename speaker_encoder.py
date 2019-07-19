import torch.nn as nn

class fc_elu(nn.Modules):
    def __init__(self, in_dim, hidden_units):
        self.fc = nn.Linear(in_dim, hidden_units)
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.fc(x)
        out = self.elu(out)
        return out

# Convolution on time dimension
# Input and output dimension remains the same
class conv_glu(nn.Modules):
    def __init__(self):

class sample_attention(nn.Modules):
    def __init__(self):
        self.key_proj = fc_elu()
        self.query_proj = fc_elu()
        self.value_proj = fc_elu()
        self.multi_head_attn = nn.MultiHeadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(d_attn, d_attn)
        self.softsign = nn.Softsign()



'''
Speaker encoder for 'Neural voice cloning'. i.e., the auxiliary network other than TTS
that is used to inference speaker embedding of unseen speaker at cloning time
'''
class SpeakerEncoder(nn.Modules):
    def __init__(self, encoder_channels, kernel_size):
        h = encoder_channels
        k = kernel_size

        self.preattention=[(h, k, 1), (h, k, 3)]
        self.convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1)]
        self.prenet_list = nn.ModuleList()
        self.convnet_list = nn.ModuleList()
        self.avg_pool = nn.AvgPool1d()
        self.sample_dim_changer = nn.Linear(f_mapped, d_embedding)
        self.sample_attn = sample_attention()

        for i in range(n_prenet):
            prenet = fc_elu()
            self.prenet_list.append(prenet)

        for i in range(n_convnet):
            convnet = conv_glu()
            self.convnet_list.append(convnet)

    # Return speaker embedding which matches the speaker embedding of MutlSpeakerTTS in dimension
    def forward(self):