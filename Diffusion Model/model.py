import torch
import torch.nn as nn

class PredictNoise(nn.Module):
    def __init__(self, in_channel = 3, 
                 hidden_channel = 256, 
                 out_channel = 3, kernel_size = 3,
                 isConditional=False, 
                 vocabSize = None,
                 device = 'cpu'):
        
        super().__init__()
        self.device = device
        self.time_dim = 128
        self.hidden_channel = hidden_channel

        self.isConditional = isConditional

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size),
            nn.LeakyReLU(),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size),
            nn.LeakyReLU(),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size),
            nn.LeakyReLU(),
        )

        self.block_4 = nn.Sequential(
            nn.ConvTranspose2d(2*hidden_channel, hidden_channel, kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size),
            nn.LeakyReLU(),
        )

        self.block_5 = nn.Sequential(
            nn.ConvTranspose2d(2*hidden_channel, hidden_channel, kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_channel, out_channel, kernel_size),
            nn.LeakyReLU(),
        )

        self.last_conv = nn.Conv2d(2*out_channel, out_channel, kernel_size,  padding = "same")
        self.embedding_layer = nn.Linear(self.time_dim, hidden_channel)
        if self.isConditional:
            self.label_embedding = nn.Embedding(vocabSize, embedding_dim=hidden_channel)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def encode_t(self, x, t, dims):
        t = self.embedding_layer(t).unsqueeze(2).unsqueeze(3)
        t = t.expand(x.shape[0], dims, x.shape[2], x.shape[3])
        return t

    def encode_y(self, x, y, dims):
        y = y.long()
        y = self.label_embedding(y).unsqueeze(2).unsqueeze(3)
        y = y.expand(x.shape[0], dims, x.shape[2], x.shape[3])
        return y
        
    def forward(self, x, t, y = None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        
        x1 = self.block_1(x)

        x2 = x1 + self.encode_t(x1, t, dims = self.hidden_channel)
        if self.isConditional : x2 = x2 + self.encode_y(x1, y, self.hidden_channel)

        x2 = self.block_2(x2)

        x3 = x2 + self.encode_t(x2, t, dims = self.hidden_channel)
        if self.isConditional : x3 = x3 + self.encode_y(x2, y, self.hidden_channel)
        
        x3 = self.block_3(x3)

        x4 = self.block_4(torch.cat((x3, x2), axis = 1))
        x5 = self.block_5(torch.cat((x4, x1), axis = 1))
        out = self.last_conv(torch.cat((x5, x), axis = 1))

        return out
