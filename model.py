import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================
# Basic CNN Blocks (unchanged)
# =========================================================
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )

# =========================================================
# CNN Encoder (unchanged)
# =========================================================
class TinyConvEncoder(nn.Module):
    def __init__(self, in_ch=4, channels=[16, 32, 64, 128]):
        super().__init__()
        self.inc = conv_block(in_ch, channels[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), conv_block(channels[0], channels[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), conv_block(channels[1], channels[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), conv_block(channels[2], channels[3]))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return [x1, x2, x3, x4]

# =========================================================
# Vision Transformer with Positional Encoding (improved)
# =========================================================
class TinyViT(nn.Module):
    def __init__(self,
                 in_ch=4,
                 patch_size=8,
                 embed_dim=48,
                 num_layers=1,
                 num_heads=4,
                 dropout=0.1):                 # Added dropout for regularization
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable positional encoding for the sequence length (Hp * Wp)
        # We'll create it dynamically in forward based on input size
        self.pos_embedding = nn.ParameterDict()  # Will be created on-the-fly

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,                     # Added dropout
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.grid_proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)                          # (B, embed_dim, Hp, Wp)
        _, D, Hp, Wp = x.shape

        # Create positional encoding if not already cached for this size
        pos_key = f"{Hp}_{Wp}"
        if pos_key not in self.pos_embedding:
            # Standard 1D learnable positional embedding
            num_patches = Hp * Wp
            pos_embed = nn.Parameter(torch.randn(1, num_patches, D) * 0.02)
            self.pos_embedding[pos_key] = pos_embed.to(x.device)
        else:
            pos_embed = self.pos_embedding[pos_key]

        # Flatten patches and add positional encoding
        tokens = x.flatten(2).permute(0, 2, 1)     # (B, num_patches, D)
        tokens = tokens + pos_embed

        tokens = self.transformer(tokens)

        # Reshape back to grid
        grid = tokens.permute(0, 2, 1).contiguous().view(B, D, Hp, Wp)
        grid = self.grid_proj(grid)
        return tokens, grid

# =========================================================
# Attention Modules (unchanged)
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx = x.max(1, keepdim=True)[0]
        a = torch.cat([avg, mx], dim=1)
        w = torch.sigmoid(self.conv(a))
        return x * w

# =========================================================
# Cross Attention (with configurable projection dim)
# =========================================================
class CrossAttention2d(nn.Module):
    def __init__(self, conv_ch, vit_dim, proj_dim=64):
        super().__init__()
        self.q = nn.Conv2d(conv_ch, proj_dim, 1)
        self.k = nn.Linear(vit_dim, proj_dim)
        self.v = nn.Linear(vit_dim, proj_dim)
        self.scale = proj_dim ** 0.5
        self.out = nn.Conv2d(proj_dim, conv_ch, 1)

    def forward(self, conv_feat, vit_feat):
        B, C, H, W = conv_feat.shape
        q = self.q(conv_feat)                           # (B, proj_dim, H, W)
        d = q.shape[1]
        q = q.view(B, d, -1).permute(0, 2, 1)           # (B, H*W, proj_dim)

        # Reshape vit_feat if it's a grid (4D)
        if vit_feat.ndim == 4:
            B2, D, Ht, Wt = vit_feat.shape
            vit = vit_feat.view(B2, D, -1).permute(0, 2, 1)  # (B, Ht*Wt, D)
        else:
            vit = vit_feat

        k = self.k(vit)                                  # (B, num_tokens, proj_dim)
        v = self.v(vit)                                  # (B, num_tokens, proj_dim)

        attn = torch.bmm(q, k.permute(0, 2, 1)) / self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)                         # (B, H*W, proj_dim)
        out = out.permute(0, 2, 1).contiguous().view(B, d, H, W)
        out = self.out(out)                              # (B, C, H, W)
        return out

# =========================================================
# Fusion Block (unchanged, but proj_dim configurable via HCSAF)
# =========================================================
class FusionBlock(nn.Module):
    def __init__(self, conv_ch, vit_dim, proj_dim=64):
        super().__init__()
        self.cross = CrossAttention2d(conv_ch, vit_dim, proj_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(conv_ch * 2, conv_ch, 3, padding=1),
            nn.BatchNorm2d(conv_ch),
            nn.GELU(),
            nn.Conv2d(conv_ch, conv_ch, 3, padding=1),
            nn.BatchNorm2d(conv_ch),
            nn.GELU()
        )
        self.se = SEBlock(conv_ch)
        self.spatial = SpatialAttention()

    def forward(self, conv_feat, vit_feat):
        attn = self.cross(conv_feat, vit_feat)
        x = torch.cat([conv_feat, attn], dim=1)
        x = self.refine(x)
        x = self.se(x)
        x = self.spatial(x)
        return x

# =========================================================
# HCSAF Module (with configurable proj_dim)
# =========================================================
class HCSAF(nn.Module):
    def __init__(self, conv_channels, vit_dim, proj_dim=64):
        super().__init__()
        self.fusers = nn.ModuleList([FusionBlock(c, vit_dim, proj_dim) for c in conv_channels])
        self.projectors = nn.ModuleList([nn.Conv2d(c, proj_dim, 1) for c in conv_channels])
        self.alpha = nn.Parameter(torch.ones(len(conv_channels)))

    def forward(self, conv_feats, vit_feat):
        fused = []
        for i, (f, fuser) in enumerate(zip(conv_feats, self.fusers)):
            x = fuser(f, vit_feat)
            x = self.projectors[i](x)
            fused.append(x)

        # Target size = size of the first (highest resolution) fused feature
        target = fused[0].shape[2:]
        ups = []
        for f in fused:
            if f.shape[2:] != target:
                f = F.interpolate(f, size=target, mode='bilinear', align_corners=False)
            ups.append(f)

        stack = torch.stack(ups)                     # (num_levels, B, proj_dim, H, W)
        w = torch.softmax(self.alpha, dim=0).view(len(ups), 1, 1, 1, 1)
        out = (w * stack).sum(0)
        return out

# =========================================================
# Attention Gate (fixed channel handling)
# =========================================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.Wg = nn.Conv2d(F_g, F_int, 1)
        self.Wx = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        # Resize g to match x spatial dimensions
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = self.psi(g1 + x1)
        return x * psi

# =========================================================
# Decoder Blocks (unchanged)
# =========================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# =========================================================
# Decoder with Attention Gates (fixed channel dimensions)
# =========================================================
class HybridDecoder(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Attention gates with correct channel numbers:
        # att3: gating from HCSAF (64 ch), skip x3 (64 ch)
        self.att3 = AttentionGate(64, 64, 32)
        # att2: gating from block3 output (64 ch), skip x2 (32 ch)
        self.att2 = AttentionGate(64, 32, 24)
        # att1: gating from block2 output (48 ch), skip x1 (16 ch)
        self.att1 = AttentionGate(48, 16, 16)

        self.block3 = DecoderBlock(64, 64, 64)   # in_ch from HCSAF (64) + skip (64) -> out 64
        self.block2 = DecoderBlock(64, 32, 48)   # in_ch from block3 (64) + skip (32) -> out 48
        self.block1 = DecoderBlock(48, 16, 32)   # in_ch from block2 (48) + skip (16) -> out 32

        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, hcsaf_out, conv_feats):
        x1, x2, x3, _ = conv_feats   # x4 is not used directly
        x = hcsaf_out                 # (B, 64, H, W)

        # Stage 3
        skip = self.att3(x3, x)       # skip is x3 modulated by x
        x = self.block3(x, skip)      # x becomes (B, 64, H/4, W/4)

        # Stage 2
        skip = self.att2(x2, x)       # skip is x2 modulated by x
        x = self.block2(x, skip)      # x becomes (B, 48, H/2, W/2)

        # Stage 1
        skip = self.att1(x1, x)       # skip is x1 modulated by x
        x = self.block1(x, skip)      # x becomes (B, 32, H, W)

        out = self.final(x)
        return out

# =========================================================
# Final Model (with input size check)
# =========================================================
class HybridHCSATUNet(nn.Module):
    def __init__(self, in_ch=4, num_classes=4, vit_dropout=0.1, proj_dim=64):
        super().__init__()
        conv_channels = [16, 32, 64, 128]

        self.encoder = TinyConvEncoder(in_ch, conv_channels)
        self.vit = TinyViT(in_ch, patch_size=8, embed_dim=48, num_layers=1,
                           num_heads=4, dropout=vit_dropout)
        self.hcsaf = HCSAF(conv_channels, vit_dim=48, proj_dim=proj_dim)
        self.decoder = HybridDecoder(num_classes)

        # Store input size for optional padding (not used in forward, but can be checked)
        self.patch_size = 8

    def forward(self, x):
        # Optional: pad input to multiples of patch_size to avoid dimension mismatches
        B, C, H, W = x.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        conv_feats = self.encoder(x)
        tokens, grid = self.vit(x)
        fused = self.hcsaf(conv_feats, grid)
        out = self.decoder(fused, conv_feats)

        # Crop back to original size if padded
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return out
