import torch
import torch.nn as nn

from deepvoice3_pytorch.modules import Conv1dGLU

class FC_ELU(nn.Module):
    def __init__(self, in_dim, hidden_units):
        super(FC_ELU, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_units)
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.fc(x)
        out = self.elu(out)
        return out

class SampleAttention(nn.Module):
    def __init__(self, d_embedding, num_heads, d_attn, f_mapped):
        super(SampleAttention, self).__init__()
        self.key_proj = FC_ELU(f_mapped, d_attn)
        self.query_proj = FC_ELU(f_mapped, d_attn)
        self.value_proj = FC_ELU(f_mapped, d_attn)
        #self.multi_head_attn = nn.MultiheadAttention(d_embedding, num_heads)
        self.fc = nn.Linear(d_attn, 1)
        self.softsign = nn.Softsign()

    def forward(self, x):
        key = self.key_proj(x)
        query = self.query_proj(x)
        value = self.value_proj(x)

        #TODO: MultiheadAttention implementation
        #x, _ = self.multi_head_attn(key, query, value)
        #x = self.fc(x)
        x = self.fc(value).squeeze(dim=-1)

        # TODO: Try normalizing with softsign and functional.normalize and compare with softmax
        # x = self.softsign(x)
        #x = nn.functional.normalize(x, dim=1)

        # I am replacing normalize with softmax -sunghee
        x = nn.functional.softmax(x, dim=1)
        return x

'''
Speaker encoder for 'Neural voice cloning'. i.e., the auxiliary network other than TTS
that is used to inference speaker embedding of unseen speaker at cloning time
'''
class SpeakerEncoder(nn.Module):
    def __init__(self, in_channels, encoder_channels, kernel_size, f_mapped, d_embedding, num_heads, d_attn,
                 dropout, batch_size, cloning_sample_size):
        super(SpeakerEncoder, self).__init__()
        h = encoder_channels
        k = kernel_size
        self.f_mapped = f_mapped
        self.batch_size = batch_size
        self.cloning_sample_size = cloning_sample_size

        self.preattention=[(h, k, 1), (h, k, 3)]
        self.convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1)]
        self.prenet_list = nn.ModuleList()
        self.convnet_list = nn.ModuleList()
        self.avg_pool = nn.AvgPool1d(kernel_size=k)
        self.cloning_sample_prj = nn.Linear(f_mapped, d_embedding)
        self.sample_attn = SampleAttention(d_embedding, num_heads, d_attn, f_mapped)

        for out_channels, kernel_size, dilation in self.preattention:
            if in_channels != f_mapped:
                self.prenet_list.append(
                    FC_ELU(in_channels, f_mapped)
                )
                in_channels = f_mapped
            else:
                self.prenet_list.append(
                    FC_ELU(f_mapped, f_mapped)
                )

        std_mul = 1.0
        for out_channels, kernel_size, dilation in self.convolutions:
            convnet = Conv1dGLU(0, None, f_mapped, f_mapped, kernel_size, dropout, dilation=dilation,
                                causal=False, residual=True, std_mul=std_mul)
            self.convnet_list.append(convnet)
            in_channels = f_mapped
            std_mul = 4.0

    # Return speaker embedding which matches the speaker embedding of MutlSpeakerTTS in dimension
    def forward(self, mel):
        # input mel: [B x N, T, F]
        x = mel.contiguous()
        # Fix dimension if necessary for FC
        # size = x.size()
        # x = x.view(-1, size[-1])
        for prenet in self.prenet_list:
            x = prenet(x)

        # [N, C, T] for conv1d
        x = x.transpose(1, 2)
        # Noncausal Conv1dGLU
        for c in self.convnet_list:
            x = c(x, None)

        mean_pool_output = nn.functional.avg_pool1d(x, x.size()[-1])
        mean_pool_output = mean_pool_output.view(-1, self.cloning_sample_size, self.f_mapped)
        attn_weights = self.sample_attn(mean_pool_output)
        cloning_samples = self.cloning_sample_prj(mean_pool_output)

        speaker_embeddings = []
        for attn, sample in zip(attn_weights, cloning_samples):
            speaker_embedding = attn.matmul(sample)
            speaker_embedding = speaker_embedding.squeeze()
            speaker_embeddings.append(speaker_embedding)
        speaker_embeddings = torch.stack(speaker_embeddings)
        return speaker_embeddings
