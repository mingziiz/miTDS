
import torch

import torch.nn as nn

from transformers import BertModel, logging



class miTDS(nn.Module):
    """ miTDS for microRNA target prediction """

    def __init__(self, model_cfg, with_esa, dropout_rate):
        super(miTDS, self).__init__()
        if not with_esa:
            self.in_channels, in_length = 8, 40
        else:
            self.in_channels, in_length = 10, 50
        self.seq_length = 68

        file_path = './pretrained_models/{}-new-12w-0'.format(model_cfg.kmer)

        self.bert = BertModel.from_pretrained(file_path, output_hidden_states=True)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            # param.retain_grad()
        self.embedding_dim = self.bert.config.hidden_size
        self.downsample = downsample(in_length=self.seq_length, embedding_dim=self.embedding_dim, out_channels=model_cfg.ds_out_channels)
        self.lstm = lstm_block(hidden_size=model_cfg.lstm_hidden_size, num_layers=model_cfg.lstm_num_layers)
        self.inception = InceptionWithAttention(10, 16, (16, 32), (8, 16), 12)
        self.cnn1 = cnn_block1(in_length=68, embedding_size=768)
        self.cnn2 = cnn_block2(in_length=33, embedding_size=128)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fcicl = nn.Linear(5016,1)


    def forward(self, input, input_ids, attention_masks, token_type_ids):
        x = self.bert(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)[0]
        dx = self.downsample(x)
        cx = self.cnn1(x)
        cx = self.cnn2(cx)
        ix = self.inception(input)
        lx = self.lstm(dx)
        iclx = torch.cat((ix,cx,lx), dim=1)
        out = self.dropout(iclx)
        out = self.fcicl(out)

        return out

class downsample(nn.Module):
    def __init__(self, in_length=None, embedding_dim=None, out_channels=None):
        super(downsample, self).__init__()
        self.layer_norm = nn.LayerNorm([in_length, embedding_dim])
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self,x):
        x = self.layer_norm(x)
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)


        return x

class lstm_block(nn.Module):
    def __init__(self, hidden_size=None, num_layers=None):
        super(lstm_block, self).__init__()

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=0.3,
                            bidirectional=True,
                            batch_first=True)
        self.cov = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=2, stride=2)
        self.ln = nn.LayerNorm(([68, 128]))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, padding=2)
        self.flatten = nn.Flatten()


    def forward(self,x):
        lstm_out = self.lstm(x)[0]
        out = self.ln(lstm_out)
        out = out.permute(0, 2, 1)
        out = self.cov(out) #[64,34]
        out = out.permute(0, 2, 1)
        out = self.flatten(out)

        return out


class cnn_block1(nn.Module):
    def __init__(self, in_length=None, embedding_size=None):
        super(cnn_block1, self).__init__()
        self.layer_norm = nn.LayerNorm([in_length, embedding_size])
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool1d(kernel_size=2)

    def forward(self,x):
        x = self.layer_norm(x)
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = x.permute(0, 2, 1)

        return x


class cnn_block2(nn.Module):
    def __init__(self, in_length=None, embedding_size=None):
        super(cnn_block2, self).__init__()
        self.layer_norm = nn.LayerNorm([in_length, embedding_size])
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=256, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8448,128)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class InceptionWithAttention(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionWithAttention, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            AttentionConvBlock(in_channels, c2[0], kernel_size=1),
            AttentionConvBlock(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            AttentionConvBlock(in_channels, c3[0], kernel_size=1),
            AttentionConvBlock(c3[0], c3[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, c4, kernel_size=1),
            nn.ReLU()
        )
        self.ca = ChannelAttention(c1 + c2[1] + c3[1] + c4)
        self.sa = SpatialAttention()
        self.flatten = nn.Flatten()

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        out = torch.cat((p1, p2, p3, p4), dim=1)
        out = self.ca(out)
        out = self.sa(out)
        out = self.flatten(out)

        return out


class AttentionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(AttentionConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.attention = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        attention_weights = self.attention(out)
        out = out * attention_weights
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x












