import torch
import torch.nn as nn
import math

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

'''
Query[t_q, d_k]
Key[t_k, d_k]
Value[t_k, d_v]
'''
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, key, query, value):
        y = torch.matmul(query, key.transpose())
        scaler = 1/math.sqrt(key.size()[1])
        y = scaler * y
        y = nn.softmax(y)
        y = y.matmul(value)
        return y

class MultiheadAttention(nn.Module):
    def __init__(self, d_attn, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_attn = d_attn
        self.d_k = int(d_attn / num_heads)
        self.num_heads = num_heads

        self.query_projections = nn.ModuleList()
        self.key_projections = nn.ModuleList()
        self.value_projections = nn.ModuleList()
        self.out_projection = nn.Linear(d_attn, d_attn)

        for i in range(num_heads):
            prj = nn.Linear(self.d_attn, self.d_k)
            self.query_projections.append(prj)

        for i in range(num_heads):
            prj = nn.Linear(self.d_attn, self.d_k)
            self.key_projections.append(prj)

        for i in range(num_heads):
            prj = nn.Linear(self.d_attn, self.d_k)
            self.value_projections.append(prj)

    def forward(self, K, Q, V):
        concat = []
        batch = K.size()[0]
        clone = K.size()[1]
        for k, q, v in zip(self.key_projections, self.query_projections, self.value_projections):
            key = k(K)
            query = q(Q)
            value = v(V)
            key_t = key.transpose(1, 2)
            y = torch.matmul(query, key_t)

            scaler = 1 / math.sqrt(key.size()[-1])
            scaled = scaler * y
            attn_weight = nn.functional.softmax(scaled, dim=-1)
            head = torch.matmul(attn_weight, value)
            concat.append(head)

        concat = torch.stack(concat, dim=3).view(batch, clone, -1)
        out = self.out_projection(concat)
        return out

class SimpleMultiheadAttention(nn.Module):
    def __init__(self, d_x, d_attn, num_heads):
        super(SimpleMultiheadAttention, self).__init__()
        self.single_head_attn = nn.Linear(d_x, d_attn)
        self.multi_head_attn = nn.Linear(d_attn, num_heads)

    def forward(self, x):
        y = self.single_head_attn(x)
        nn.functional.relu(y)
        y = self.multi_head_attn(y)
        y = nn.functional.softmax(y)
        return y

class SampleAttention(nn.Module):
    def __init__(self, d_embedding, num_heads, d_attn, f_mapped):
        super(SampleAttention, self).__init__()
        self.key_proj = FC_ELU(f_mapped, d_attn)
        self.query_proj = FC_ELU(f_mapped, d_attn)
        self.value_proj = FC_ELU(f_mapped, d_attn)
        self.multi_head_attn = MultiheadAttention(d_attn, num_heads)
        self.fc = nn.Linear(d_attn, 1)
        self.softsign = nn.Softsign()

    def forward(self, x):
        key = self.key_proj(x)
        query = self.query_proj(x)
        value = self.value_proj(x)

        x = self.multi_head_attn(key, query, value)
        x = self.fc(x).squeeze(dim=-1)

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

        self.preattention = 2
        self.convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1)]
        self.prenet_list = nn.ModuleList()
        self.convnet_list = nn.ModuleList()
        self.avg_pool = nn.AvgPool1d(kernel_size=k)
        self.cloning_sample_prj = nn.Linear(f_mapped, d_embedding)
        self.temporal_attention = SampleAttention(d_embedding, num_heads, d_attn, f_mapped)
        self.sample_attn = SampleAttention(d_embedding, num_heads, d_attn, f_mapped)

        for i in range(self.preattention):
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
        for prenet in self.prenet_list:
            x = prenet(x)

        # [N, C, T] for conv1d
        x = x.transpose(1, 2)
        # Noncausal Conv1dGLU
        for c in self.convnet_list:
            x = c(x, None)

        # Temporal aggregation
        temporal_attn_weights = self.temporal_attention(x.transpose(1, 2)) # [b x N, T]

        sample_embeddings = []
        for attn, frame in zip(temporal_attn_weights, x):
            sample_embedding = attn.matmul(frame.transpose(0, 1))
            # sample_embedding = sample_embedding.squeeze()
            sample_embeddings.append(sample_embedding)
        sample_embeddings = torch.stack(sample_embeddings)

        time_aggregated = sample_embeddings.view(-1, self.cloning_sample_size, self.f_mapped)
        attn_weights = self.sample_attn(time_aggregated)
        cloning_samples = self.cloning_sample_prj(time_aggregated)

        speaker_embeddings = []
        for attn, sample in zip(attn_weights, cloning_samples):
            speaker_embedding = attn.matmul(sample)
            # speaker_embedding = speaker_embedding.squeeze()
            speaker_embeddings.append(speaker_embedding)
        speaker_embeddings = torch.stack(speaker_embeddings)
        return speaker_embeddings
