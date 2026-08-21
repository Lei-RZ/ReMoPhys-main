import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _, _ = x.shape
        weight = self.avg_pool(x).view(batch_size, channels)
        weight = self.fc(weight).view(batch_size, channels, 1, 1, 1)
        return x * weight


class TemporalDiffConv3d(nn.Module):
    """3D convolution with a temporal-difference residual input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
        theta: float = 0.2,
    ):
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.theta == 0 or x.size(2) <= 1:
            return self.conv(x)

        temporal_difference = torch.zeros_like(x)
        temporal_difference[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        return self.conv(x + self.theta * temporal_difference)


def temporal_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size,
    padding,
    theta: float,
) -> nn.Sequential:
    return nn.Sequential(
        TemporalDiffConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            theta=theta,
        ),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        SEBlock(out_channels),
    )


class IR_SE_CNN(nn.Module):
    """Infrared feature encoder used by the dual-modal path."""

    def __init__(self, input_channels: int = 3, theta: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            TemporalDiffConv3d(
                input_channels, 16, kernel_size=3, padding=1, theta=theta
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            SEBlock(16),
            nn.MaxPool3d(2),

            TemporalDiffConv3d(
                16, 32, kernel_size=3, padding=1, theta=theta
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool3d(2),

            TemporalDiffConv3d(
                32, 64, kernel_size=3, padding=1, theta=theta
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

class CrossSpectrumAdaptiveAggregation(nn.Module):
    def __init__(self, feature_channels: int = 64):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv3d(
                feature_channels * 2,   # 原来是 * 4
                feature_channels,
                kernel_size=1,
            ),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                feature_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                32,
                2,
                kernel_size=1,
            ),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
    ) -> torch.Tensor:

        if x_rgb.shape != x_ir.shape:
            x_ir = F.interpolate(
                x_ir,
                size=x_rgb.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        fusion_input = torch.cat(
            [
                x_rgb,
                x_ir,
            ],
            dim=1,
        )

        modality_weight = torch.softmax(
            self.gate(fusion_input),
            dim=1,
        )

        fused = (
            modality_weight[:, 0:1] * x_rgb
            + modality_weight[:, 1:2] * x_ir
        )

        refined = self.refine(fused)

        return fused + refined

# ---------------------------------------------------------------------
# Periodic signal heads
# ---------------------------------------------------------------------
class MultiScalePeriodicTemporalMixer(nn.Module):
    """Multi-scale depthwise temporal mixer for periodic signals."""

    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        dilations=(1, 2, 4, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=dilation * (kernel_size // 2),
                        dilation=dilation,
                        groups=channels,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.ELU(),
                )
                for dilation in dilations
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(
                channels * len(dilations),
                channels,
                kernel_size=1,
            ),
            nn.BatchNorm1d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat(
            [branch(x) for branch in self.branches],
            dim=1,
        )
        return x + self.mix(mixed)


class PeriodicSignalHead(nn.Module):
    def __init__(self, frames: int, channels: int = 64):
        super().__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose3d(
                channels,
                channels,
                kernel_size=[4, 1, 1],
                stride=[2, 1, 1],
                padding=[1, 0, 0],
            ),
            nn.BatchNorm3d(channels),
            nn.ELU(),
            nn.ConvTranspose3d(
                channels,
                channels,
                kernel_size=[4, 1, 1],
                stride=[2, 1, 1],
                padding=[1, 0, 0],
            ),
            nn.BatchNorm3d(channels),
            nn.ELU(),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
        )
        self.mixer = MultiScalePeriodicTemporalMixer(
            channels=channels
        )
        self.out = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.decode(x).squeeze(-1).squeeze(-1)
        feature = self.mixer(feature)
        return self.out(feature).squeeze(1)


# ---------------------------------------------------------------------
# Task-specific adapters
# ---------------------------------------------------------------------
class TaskMoEAdapter(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        rank: int = 16,
        num_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts

        hidden = max(channels // 2, 8)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        channels,
                        rank,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(
                        rank,
                        channels,
                        kernel_size=1,
                        bias=False,
                    ),
                )
                for _ in range(num_experts)
            ]
        )
        self.norm = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_weight = torch.softmax(
            self.router(x),
            dim=1,
        )
        expert_output = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1,
        )
        adapted = (
            expert_weight[:, :, None, None, None, None]
            * expert_output
        ).sum(dim=1)
        return self.norm(x + adapted)


# ---------------------------------------------------------------------
# SpO2 head
# ---------------------------------------------------------------------
class LegacySpO2Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                1,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                16,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, rppg: torch.Tensor) -> torch.Tensor:
        spo2 = self.net(rppg.unsqueeze(1))
        return spo2 * 15.0 + 85.0



# class RatioOfRatiosSpO2Head(nn.Module):
#     def __init__(self, feature_channels: int = 64, hidden: int = 96):
#         super().__init__()
#         input_dim = 24 + 12 + 3 + feature_channels
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.LayerNorm(hidden),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.15),
#             nn.Linear(hidden, hidden // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden // 2, 1),
#             nn.Sigmoid(),
#         )

#     @staticmethod
#     def _trace_stats(trace: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#         mean = trace.mean(dim=-1)
#         centered = trace - mean.unsqueeze(-1)
#         std = centered.std(dim=-1, unbiased=False)
#         abs_mean = trace.abs().mean(dim=-1)
#         rms = torch.sqrt((trace ** 2).mean(dim=-1) + eps)
#         return torch.cat([mean, std, abs_mean, rms], dim=1)

#     def forward(
#         self,
#         x_rgb: torch.Tensor,
#         x_ir: torch.Tensor,
#         fused_feat: torch.Tensor,
#         rppg: torch.Tensor,
#     ) -> torch.Tensor:
#         eps = 1e-6

#         rgb_trace = x_rgb.mean(dim=(-1, -2))
#         if x_ir is None:
#             ir_trace = torch.zeros_like(rgb_trace)
#         else:
#             ir_trace = x_ir.mean(dim=(-1, -2))
#             if ir_trace.size(1) != rgb_trace.size(1):
#                 if ir_trace.size(1) == 1:
#                     ir_trace = ir_trace.repeat(1, 3, 1)
#                 else:
#                     ir_trace = ir_trace[:, :3]

#         rgb_trace = rgb_trace[:, :3]
#         if rgb_trace.size(1) < 3:
#             rgb_trace = F.pad(
#                 rgb_trace,
#                 (0, 0, 0, 3 - rgb_trace.size(1)),
#             )

#         ir_trace = ir_trace[:, :3]
#         if ir_trace.size(1) < 3:
#             ir_trace = F.pad(
#                 ir_trace,
#                 (0, 0, 0, 3 - ir_trace.size(1)),
#             )

#         rgb_mean = rgb_trace.mean(dim=-1)
#         ir_mean = ir_trace.mean(dim=-1)
#         rgb_std = (
#             rgb_trace - rgb_mean.unsqueeze(-1)
#         ).std(dim=-1, unbiased=False)
#         ir_std = (
#             ir_trace - ir_mean.unsqueeze(-1)
#         ).std(dim=-1, unbiased=False)

#         modality_stats = torch.cat(
#             [
#                 self._trace_stats(rgb_trace),
#                 self._trace_stats(ir_trace),
#             ],
#             dim=1,
#         )

#         cross_ratio = torch.cat(
#             [
#                 rgb_std / (ir_std + eps),
#                 rgb_mean.abs() / (ir_mean.abs() + eps),
#                 rgb_std[:, 0:1] / (rgb_std[:, 1:2] + eps),
#                 rgb_std[:, 0:1] / (rgb_std[:, 2:3] + eps),
#                 rgb_std[:, 1:2] / (rgb_std[:, 2:3] + eps),
#                 ir_std[:, 0:1] / (ir_std[:, 1:2] + eps),
#                 ir_std[:, 0:1] / (ir_std[:, 2:3] + eps),
#                 ir_std[:, 1:2] / (ir_std[:, 2:3] + eps),
#             ],
#             dim=1,
#         )

#         rppg_centered = rppg - rppg.mean(dim=-1, keepdim=True)
#         rppg_stats = torch.cat(
#             [
#                 rppg.mean(dim=-1, keepdim=True),
#                 rppg_centered.std(
#                     dim=-1,
#                     keepdim=True,
#                     unbiased=False,
#                 ),
#                 torch.sqrt(
#                     (rppg ** 2).mean(dim=-1, keepdim=True) + eps
#                 ),
#             ],
#             dim=1,
#         )

#         deep_feature = fused_feat.mean(dim=(2, 3, 4))
#         feat = torch.cat(
#             [
#                 modality_stats,
#                 cross_ratio,
#                 rppg_stats,
#                 deep_feature,
#             ],
#             dim=1,
#         )
#         feat = torch.nan_to_num(
#             feat,
#             nan=0.0,
#             posinf=1e3,
#             neginf=-1e3,
#         )

#         return self.mlp(feat) * 16.5 + 83.5


# ---------------------------------------------------------------------
# Complete ReMoPhysNet model
# ---------------------------------------------------------------------
class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    """Complete ReMoPhysNet model.

    The class name is retained for compatibility with the original trainer.
    """

    def __init__(
        self,
        frames: int = 128,
        temporal_diff_theta: float = 0.2,
    ):
        super().__init__()
        self.frames = frames
        theta = temporal_diff_theta

        self.ConvBlock1 = temporal_conv_block(
            3, 16, [1, 5, 5], [0, 2, 2], theta
        )
        self.ConvBlock2 = temporal_conv_block(
            16, 32, [3, 3, 3], 1, theta
        )
        self.ConvBlock3 = temporal_conv_block(
            32, 64, [3, 3, 3], 1, theta
        )
        self.ConvBlock4 = temporal_conv_block(
            64, 64, [3, 3, 3], 1, theta
        )
        self.ConvBlock5 = temporal_conv_block(
            64, 64, [3, 3, 3], 1, theta
        )
        self.ConvBlock6 = temporal_conv_block(
            64, 64, [3, 3, 3], 1, theta
        )
        self.ConvBlock7 = temporal_conv_block(
            64, 64, [3, 3, 3], 1, theta
        )
        self.ConvBlock8 = temporal_conv_block(
            64, 64, [3, 3, 3], 1, theta
        )

        self.MaxpoolSpa = nn.MaxPool3d(
            (1, 2, 2),
            stride=(1, 2, 2),
        )
        self.MaxpoolSpaTem = nn.MaxPool3d(
            (2, 2, 2),
            stride=2,
        )

        self.ir_encoder = IR_SE_CNN(
            input_channels=3,
            theta=theta,
        )
        self.fusion_net = CrossSpectrumAdaptiveAggregation(
            feature_channels=64
        )

        self.bvp_adapter = TaskMoEAdapter(
            channels=64
        )
        self.rr_adapter = TaskMoEAdapter(
            channels=64
        )

        self.rppg_branch = PeriodicSignalHead(
            frames=frames,
            channels=64,
        )
        self.rr_branch = PeriodicSignalHead(
            frames=frames,
            channels=64,
        )
        self.legacy_spo2_head = LegacySpO2Head()

    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """Single-modal encoder path."""

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)

        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)

        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock8(x)
        return x

    def share_m(self, x: torch.Tensor) -> torch.Tensor:
        """RGB encoder path used before RGB/IR fusion."""

        _, _, input_frames, input_height, input_width = x.shape
        target_size = (
            max(1, input_frames // 4),
            max(1, input_height // 4),
            max(1, input_width // 4),
        )

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)

        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)

        x = F.adaptive_avg_pool3d(
            x,
            target_size,
        )
        return x

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor = None,
    ):
        input_length = x1.size(2)

        if x2 is not None:
            rgb_feature = self.share_m(x1)
            ir_feature = self.ir_encoder(x2)
            shared_feature = self.fusion_net(
                rgb_feature,
                ir_feature,
            )
        else:
            shared_feature = self.encode_video(x1)

        bvp_feature = self.bvp_adapter(shared_feature)
        rr_feature = self.rr_adapter(shared_feature)

        rppg = self.rppg_branch(bvp_feature)
        rr = self.rr_branch(rr_feature)

        if rppg.size(1) != input_length:
            rppg = F.interpolate(
                rppg.unsqueeze(1),
                size=input_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        if rr.size(1) != input_length:
            rr = F.interpolate(
                rr.unsqueeze(1),
                size=input_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        spo2 = self.legacy_spo2_head(rppg)
        return rppg, spo2, rr


# Optional descriptive alias. The trainer-compatible class above remains primary.
ReMoPhysNet = PhysNet_padding_Encoder_Decoder_MAX
