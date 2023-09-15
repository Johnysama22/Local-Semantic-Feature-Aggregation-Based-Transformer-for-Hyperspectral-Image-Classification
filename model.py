import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)
        self.sr = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, mask=None, use_SR=False):
        b, n, d, h = *x.shape, self.heads
        s = int((n-1) ** 0.5)
        c = x[:,0,:].reshape(b,1,d)
        f = x[:,1:,:]

        # sr k, v
        if use_SR==True:
            q = x.reshape(b, n, h, d // h).permute(0, 2, 1, 3) # 64, 8, 65, 8
            f_ = f.permute(0, 2, 1).reshape(b, d, s, s)
            f_ = self.sr(f_ )
            f_ = rearrange(f_, 'b h n d -> b h (n d)').permute(0, 2, 1) # .reshape(b, C, -1).permute(0, 2, 1)
            f_ = torch.cat((c, f_), dim=1)
            f_ = self.norm(f_)
            f_ = self.act(f_)
            kv = self.to_kv(f_).chunk(2, dim = -1)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        else:
            qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x

BATCH_SIZE_TRAIN = 64
NUM_CLASS = 9
# embedding dimension
dim1 = 32
dim2 = 64
dim3 = 128
class LSFAT(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(LSFAT, self).__init__()
        self.L = num_tokens
        # self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=32, kernel_size=(3, 8, 8)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.patch_to_embedding1 = nn.Sequential(
            nn.Linear(32*28, dim1),
            nn.LayerNorm(dim1),
        )
        self.patch_to_embedding2 = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.LayerNorm(dim2),
        )
        self.patch_to_embedding3 = nn.Sequential(
            nn.Linear(dim2, dim3),
            nn.LayerNorm(dim3),
        )

        self.pos_embedding1 = nn.Parameter(torch.empty(1, 64 + 1, dim1))
        torch.nn.init.normal_(self.pos_embedding1, std=.02)

        self.pos_embedding2 = nn.Parameter(torch.empty(1, (16 + 1), dim2))
        torch.nn.init.normal_(self.pos_embedding2, std=.02)

        self.pos_embedding3 = nn.Parameter(torch.empty(1, (4 + 1), dim3))
        torch.nn.init.normal_(self.pos_embedding3, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim1))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer1 = Transformer(dim1, depth, heads, 64, dropout)
        self.transformer2 = Transformer(dim2, depth, heads, 16, dropout)
        self.transformer3 = Transformer(dim3, depth, heads, 4, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim1, num_classes)
        self.nn2 = nn.Linear(dim2, num_classes)
        self.nn3 = nn.Linear(dim3, num_classes)

    def LSFAT_Layer1(self, x, mask=None):
        x = rearrange(x, 'b c h w -> b (h w) c')
        # pixel embedding
        x = self.patch_to_embedding1(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding1
        x = self.dropout(x)
        # neighborhood aggregation attention
        x = self.transformer1(x, mask)
        # separate cls token and feature token
        c = self.to_cls_token(x[:, 0]) # cls token
        x = self.to_cls_token(x[:, 1:]) # feature token
        return c, x

    def LSFAT_Layer2(self, p, c, mask=None, dim=dim1, k=0):
        p = p.reshape(p.shape[0], 8, 8, dim)
        x = torch.zeros(p.shape[0], 16, dim).cuda()
        # neighborhood aggregation-based embedding
        for i in range(0,3):
            for j in range(0,3):
                temp = p[:,2*i:2*i+2,2*j:2*j+2,:]
                temp = temp.reshape(temp.shape[0], 4, dim)
                temp = temp.mean(dim=1)
                x[:,k,:] = temp
                k += 1
        x = self.patch_to_embedding2(x)
        c = self.patch_to_embedding2(c)
        cls_tokens = c.reshape(x.shape[0],1,dim2)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding2
        x = self.dropout(x)
        # neighborhood aggregation attention
        x = self.transformer2(x, mask)
        # separate cls token and feature token
        c = self.to_cls_token(x[:, 0]) # cls token
        x = self.to_cls_token(x[:, 1:]) # feature token
        return c, x

    def LSFAT_Layer3(self, p, c, mask=None, dim=dim2, k=0):
        p = p.reshape(p.shape[0], 4, 4, dim)
        x = torch.zeros(p.shape[0], 4, dim).cuda()
        # neighborhood aggregation-based embedding
        for i in range(0,1):
            for j in range(0,1):
                temp = p[:,2*i:2*i+2,2*j:2*j+2,:]
                temp = temp.reshape(temp.shape[0], 4, dim)
                temp = temp.mean(dim=1)
                x[:,k,:] = temp
                k += 1
        x = self.patch_to_embedding3(x)
        c = self.patch_to_embedding3(c)
        cls_tokens = c.reshape(x.shape[0],1,dim3)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding3
        x = self.dropout(x)
        x = self.transformer3(x, mask)

        # separate cls token and feature token
        c = self.to_cls_token(x[:, 0]) # cls token
        x = self.to_cls_token(x[:, 1:]) # feature token
        return c, x

    def forward(self, x):
        # 3d convolution
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        # three-layer transformer
        c1, x = self.LSFAT_Layer1(x)
        c2, x = self.LSFAT_Layer2(x, c1)
        c3, x = self.LSFAT_Layer3(x, c2)

        return self.nn3(c3)

if __name__ == '__main__':
    model = LSFAT()
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 15, 15).cuda()
    y = model(input)
    print(y.size())
