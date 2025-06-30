from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple



model_urls = {
    "swin_b": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth",
    "swin_s": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth",
    "swin_t": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth"

}


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchMerging(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=2,
                 bias=False,
                 norm_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.sampler = nn.Unfold(
            kernel_size=stride, dilation=1, padding=0, stride=stride)

        sample_dim = stride**2 * in_channels

        if norm_layer is not None:
            self.norm = norm_layer(sample_dim)
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, hw_shape):
        """
        x: x.shape -> [B, H*W, C]
        hw_shape: (H, W)
        """
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        # stride is fixed to be equal to kernel_size.
        if (H % self.stride != 0) or (W % self.stride != 0):
            x = F.pad(x, (0, W % self.stride, 0, H % self.stride))

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        x = self.sampler(x)  # B, 4*C, H/2*W/2
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        x = self.norm(x) if self.norm else x
        x = self.reduction(x)

        down_hw_shape = (H + 1) // 2, (W + 1) // 2
        return x, down_hw_shape


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type=None,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 dilation=1,
                 pad_to_patch_size=True,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims

        if stride is None:
            stride = kernel_size

        self.pad_to_patch_size = pad_to_patch_size

        # The default setting of patch size is equal to kernel size.
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        elif isinstance(patch_size, tuple):
            if len(patch_size) == 1:
                patch_size = to_2tuple(patch_size[0])
            assert len(patch_size) == 2, \
                f'The size of patch should have length 1 or 2, ' \
                f'but got {len(patch_size)}'

        self.patch_size = patch_size

        # Use conv layer to embed
        #self.projection = nn.Unfold(kernel_size=(14, 14), stride=(8, 8), padding=(3, 3))
        conv_type = conv_type or 'Conv2d'
        if conv_type == 'Conv2d':
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        else:
            raise NotImplementedError

        if norm_layer is not None:
            self.norm = norm_layer(embed_dims)
        else:
            self.norm = None

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # TODO: Process overlapping op
        if self.pad_to_patch_size:
            # Modify H, W to multiple of patch size.
            if H % self.patch_size[0] != 0:
                x = F.pad(
                    x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(
                    x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))

        x = self.projection(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class FFN(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_layer=None,
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 **kwargs):
        super().__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = act_layer()

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = dropout_layer if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.
        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class WindowMSA(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.
                 ):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_init(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=None):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate)

        self.drop = dropout_layer

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1),
                                   device=query.device)  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=None,
                 norm_layer=None):

        super(SwinBlock, self).__init__()


        self.norm1 = norm_layer(embed_dims)
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=DropPath(drop_prob=drop_path_rate))

        self.norm2 = norm_layer(embed_dims)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=DropPath(drop_prob=drop_path_rate),
            act_layer=act_layer,
            add_identity=True)

    def forward(self, x, hw_shape):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)

        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)

        return x


class SwinBlockSequence(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_layer=None,
                 norm_layer=None):
        super().__init__()

        drop_path_rate = drop_path_rate if isinstance(
            drop_path_rate,
            list) else [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate[i],
                act_layer=act_layer,
                norm_layer=norm_layer)

            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class SwinTransformer(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(SwinTransformer, self).__init__()

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            pad_to_patch_size=True,
            norm_layer=norm_layer if patch_norm else None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = nn.ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_layer=norm_layer if patch_norm else None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[:depths[i]],
                downsample=downsample,
                act_layer=act_layer,
                norm_layer=norm_layer)
            self.stages.append(stage)

            dpr = dpr[depths[i]:]
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = norm_layer(self.num_features[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        self._init_weights()

    def _init_weights(self):
        if self.use_abs_pos_embed:
            trunc_normal_init(self.absolute_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)

        hw_shape = (self.patch_embed.DH, self.patch_embed.DW)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs



def swin_b(pretrained: bool = False, progress: bool = True):
    my_swin = SwinTransformer(
        embed_dims=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32)
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['swin_b'], progress=progress)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        if list(state_dict.keys())[0].startswith('backbone.'):
            state_dict_new = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    state_dict_new[k[9:]] = v
            state_dict = state_dict_new
        my_swin.load_state_dict(state_dict, strict=True)
    return my_swin


def swin_s(pretrained: bool = False, progress: bool = True):
    my_swin = SwinTransformer(
        embed_dims=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24)
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['swin_s'], progress=progress)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        if list(state_dict.keys())[0].startswith('backbone.'):
            state_dict_new = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    state_dict_new[k[9:]] = v
            state_dict = state_dict_new
        my_swin.load_state_dict(state_dict, strict=False) #strict=True
    return my_swin


def swin_t(pretrained: bool = False, progress: bool = True):
    my_swin = SwinTransformer(
        embed_dims=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24)
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['swin_t'], progress=progress)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        if list(state_dict.keys())[0].startswith('backbone.'):
            state_dict_new = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    state_dict_new[k[9:]] = v
            state_dict = state_dict_new
        my_swin.load_state_dict(state_dict, strict=False) #strict=True
    return my_swin


if __name__ == '__main__':
    model = swin_l(pretrained=True)
    # print(model)
    img = torch.rand((1, 3, 512, 512))
    feat = model(img)
    for item in feat:
        print(item.shape)

    from thop import profile

    flops, params = profile(model, inputs=(img,), verbose=False)
    print("FLOPS:", flops / 1e9)
    print("PARAMS:", params / 1e6)