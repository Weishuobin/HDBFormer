import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GFA(nn.Module):
    def __init__(self, num_head, dim):
        super().__init__()
        self.num_head = num_head
        self.pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.l = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.kv = nn.Linear(dim, dim)
        self.x_e_linear = nn.Linear(dim * 2, dim // 2)
        self.window = 6

    def forward(self, x, x_e):
        B, H, W, C = x.size()
        x_e = torch.cat([x, x_e], dim=3)
        x_e = x_e.permute(0, 3, 1, 2)

        x = self.l(x)
        x = self.act(x)
        b = x
        kv = self.kv(b)
        kv = kv.reshape(B, H * W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        x_e = self.pool(x_e).permute(0, 2, 3, 1)

        x_e = self.x_e_linear(x_e)
        x_e = x_e.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)
        m = x_e

        attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).reshape(B, self.num_head, self.window, self.window, C // self.num_head // 2).permute(0, 1, 4, 2, 3).reshape(B, C // 2, self.window, self.window)
        attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        return attn

class LFA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_cut = nn.Linear(dim, dim // 2)
        self.e_fore = nn.Linear(dim, dim // 2)
        self.e_conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=dim // 2)
        self.e_back = nn.Linear(dim // 2, dim // 2)

    def forward(self, x, x_e):
        x = self.q_cut(x)
        x_e = self.e_back(self.e_conv(self.e_fore(x_e).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        x = x * x_e
        return x

class feature_fusion_block(nn.Module):
    def __init__(self, dim, num_head=8):
        super().__init__()
        self.num_head = num_head
        self.proj = nn.Linear(dim * 3 //2, dim)
        self.proj_e = nn.Linear(dim * 3 //2, dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.GFA = GFA(self.num_head,dim)
        self.LFA = LFA(dim)

    def forward(self, x, x_e):
        x = x.transpose(1, 3).transpose(1, 2)
        x_e = x_e.transpose(1, 3).transpose(1, 2)


        x = self.norm(x)
        x_e = self.norm_e(x_e)

        gfa1 = self.GFA(x, x_e)

        lfa1 = self.LFA(x, x_e)
        lfa2 = self.LFA(x_e, x)

        x = torch.cat([ gfa1, lfa1,lfa2], dim=3)

        x_e = self.proj_e(x)
        x = self.proj(x)

        x = x.permute(0, 3, 1, 2)
        x_e = x_e.permute(0, 3, 1, 2)
        return x, x_e
