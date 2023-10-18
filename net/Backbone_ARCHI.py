from typing import Optional, Tuple, Dict, Union
import math
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from mobilevit.model_config import get_config
from mobilevit.transformer import LinearTransformerBlock
from mobilevit.transformer import GroupNorm1


class MobileViT_v2Block(nn.Module):

    def __init__(
            self,
            in_channels: int,
            transformer_dim: int,
            n_transformer_blocks: int = 2,
            patch_h: int = 8,
            patch_w: int = 8,
            conv_ksize: Optional[int] = 3,
            **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            groups=in_channels,
            **kwargs
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
            **kwargs
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_act=False,
            **kwargs
        )
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)
        self.n_transformer_blocks = n_transformer_blocks

        self.transformer = nn.Sequential(*[
            LinearTransformerBlock(
                transformer_dim,
                mlp_ratio=2.0,
                attn_drop=0.0,
                drop=0.0,
                drop_path=0.0,
                act_layer=nn.SiLU,
                norm_layer=GroupNorm1
            )
            for _ in range(n_transformer_blocks)
        ])
        self.norm = GroupNorm1(transformer_dim)
        self.conv_proj = conv_1x1_out
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        batch_size, channels, pixels, num_patches = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        fm = self.local_rep(x)

        patches, info_dict = self.unfolding(fm)
        patches = self.transformer(patches)
        patches = self.norm(patches)

        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        return fm


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Optional[Union[int, Tuple[int, int]]] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = False,
            use_norm: Optional[bool] = True,
            use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.stride = stride
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name='conv', module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name='norm', module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name='act', module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


def make_divisible(
        v: Union[float, int],
        divisor: Optional[int] = 8,
        min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: Union[int, float],
            skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()

        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                use_act=False,
                use_norm=True,
                kernel_size=1,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_rs_connect = (
                self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_rs_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViT2(nn.Module):
    def __init__(self, model_cfg: Dict):
        super(MobileViT2, self).__init__()

        image_channels = 3
        out_channels = 24
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        # in channel = 16 out channel = 16
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        # in channel = 16 out channel = 24
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        # in channel = 24 out channel = 48
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        # in channel = 48 out channel = 64
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])
        # in channel = 64 out channel = 80

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]

        block.append(MobileViT_v2Block(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            n_conv33_block=cfg.get("n_conv33_block", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)  # 3 224 224 => 16 112 112
        x = self.layer_1(x)  # 16 112 112=> 16 112 112
        x = self.layer_2(x)  # 16 112 112=> 24 56 56

        x = self.layer_3(x)  # 24 56 56=> 48 28 28
        x = self.layer_4(x)  # 48 28 28=> 64 14 14
        x = self.layer_5(x)  # 64 14 14=> 80 7 7
        return x


def mobile_vit_small():
    config = get_config("small")
    m = MobileViT2(config)
    return m
