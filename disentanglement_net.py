import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import numbers

class EmbeddingNetHyperX(nn.Module):
    def __init__(self, input_channels, n_outputs=128, patch_size=5, n_classes=None):
        super(EmbeddingNetHyperX, self).__init__()
        self.dim = 200

        # 1st conv layer
        # input [input_channels x patch_size x patch_size]
        self.convnet = nn.Sequential(
            nn.Conv2d(input_channels, self.dim, kernel_size=1, padding=0),  # input channels
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0,),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),

            nn.AvgPool2d(patch_size, stride=1)

        )

        self.n_outputs = n_outputs
        self.fc = nn.Linear(self.dim, self.n_outputs)

    def extract_features(self, x):

        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc[0](output)

        return output

    def forward(self, x):

        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)
    def output_num(self):
        return self.n_outputs


class Feature_extraction(nn.Module):
    def __init__(self):
        super(Feature_extraction, self).__init__()
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=16, padding=(0, 0, 0),kernel_size=(7, 1, 1), stride=(2, 1, 1))
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(16, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.conv12 = nn.Conv3d(in_channels=16, out_channels=16, padding=(3, 0, 0),kernel_size=(7, 1, 1), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(32, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.conv13 = nn.Conv3d(in_channels=32, out_channels=16, padding=(3, 0, 0), kernel_size=(7, 1, 1),stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=16, padding=(3, 0, 0), kernel_size=(7, 1, 1),stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(64, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.conv15 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(48, 1, 1), stride=(1, 1, 1))
        self.batch_norm_spectral = nn.Sequential(
            nn.BatchNorm3d(64, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.global_pooling = nn.AdaptiveAvgPool3d(1)

    def forward(self, X):
        X = X.unsqueeze(1)
        x11 = self.conv11(X)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        x13 = torch.cat((x11, x12), dim=1)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)
        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        x1 = self.batch_norm_spectral(x16)
        x1 = self.global_pooling(x1)
        output = x1.squeeze(-1).squeeze(-1).squeeze(-1)

        return output



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  #q torch.Size([64, 8, 32, 49])
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)  # 求通道注意力图时，对（hw）作归一化
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Attention2(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention2, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=2)  # 求通道注意力图时，对（hw）作归一化
        k = torch.nn.functional.normalize(k, dim=2)

        attn = (k.transpose(-2, -1) @ q) * self.temperature
        #attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.attn2 = Attention2(dim, num_heads, bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = x + self.attn2(self.norm3(x))
        x = x + self.ffn2(self.norm4(x))

        return x


NUM_CLASS = 7

class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, dim=256, depth=1, heads=8, ffn_expansion_factor=2,  bias =False, LayerNorm_type = 'WithBias'):
        super(SSFTTnet, self).__init__()
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*100, out_channels=256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )


        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(depth)])



        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):

        x = x.unsqueeze(1)

        x = self.conv3d_features(x)  # torch.Size([64, 8, 100, 9, 9])
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)  # torch.Size([64, 256, 7, 7])
        x = self.encoder_level1(x)  # torch.Size([64, 256, 7, 7])
        x = self.global_pooling(x)
        x = x.squeeze(-1).squeeze(-1)
        #x = self.nn1(x)

        return x

class Feature_disentangle1(nn.Module):
    def __init__(self):
        super(Feature_disentangle1, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2_fc = nn.BatchNorm1d(64)
    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


class Feature_disentangle2(nn.Module):
    def __init__(self):
        super(Feature_disentangle1, self).__init__()
        self.fc1 = nn.Linear(128, 48)
        self.bn1_fc = nn.BatchNorm1d(48)
        self.fc2 = nn.Linear(48, 32)
        self.bn2_fc = nn.BatchNorm1d(32)
    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


class Feature_discriminator(nn.Module):
    def __init__(self):
        super(Feature_discriminator, self).__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return x


class Reconstructor1(nn.Module):
    def __init__(self):
        super(Reconstructor1, self).__init__()
        self.fc = nn.Linear(128, 256)
    def forward(self,x):
        x = self.fc(x)
        return x

class Reconstructor2(nn.Module):
    def __init__(self):
        super(Reconstructor1, self).__init__()
        self.fc = nn.Linear(64, 128)
    def forward(self,x):
        x = self.fc(x)
        return x


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 32)
        self.ad_layer2 = nn.Linear(32, 32)
        self.ad_layer3 = nn.Linear(32, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x


class Mixer(nn.Module):
    def __init__(self):
        super(Mixer, self).__init__()
        self.fc1_x = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32,16)
    def forward(self, x):
        x = F.relu(self.fc1_x(x))
        x = self.fc2(x)
        return x

class Mixer2(nn.Module):
    def __init__(self):
        super(Mixer2, self).__init__()
        self.fc1_x = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32,16)
    def forward(self, x):
        x = F.relu(self.fc1_x(x))
        x = self.fc2(x)
        return x