import torch
from torch import nn
from torch.nn import functional as F
from lib.models.decoder.lama_ffc import FFCResnetBlock
from lib.models.decoder.dpt_blocks import FeatureFusionBlock_custom, Interpolate


class LaMaFeatureInpainter(nn.Module):
    def __init__(self, channels=512, blocks=18):
        super().__init__()

        self.blocks = nn.Sequential(*[FFCResnetBlock(channels, padding_type='reflect', norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, ratio_gin=0.75, ratio_gout=0.75, inline=True) for _ in range(blocks)])


    def forward(self, features):
        with torch.amp.autocast(device_type=features.device.type, enabled=False):
            result = self.blocks(features.float())

        return result.to(features.dtype)


class DPTFourHeadDecoder(nn.Module):
    def __init__(self, channels=512, features=256):
        super().__init__()

        stage_channels = [256, 512, 1024, 1024]
        resizers = [nn.ConvTranspose2d(256, 256, 4, stride=4), nn.ConvTranspose2d(512, 512, 2, stride=2), nn.Identity(), nn.Conv2d(1024, 1024, 3, stride=2, padding=1)]

        self.reassemble = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, out, 1), resize) for out, resize in zip(stage_channels, resizers)])
        self.scratch = nn.ModuleList([nn.Conv2d(out, features, 3, padding=1, bias=False) for out in stage_channels])
        self.refinenet = nn.ModuleList([FeatureFusionBlock_custom(features, nn.ReLU(False), align_corners=True) for _ in range(4)])
        del self.refinenet[3].resConfUnit1

        def rgb_head():
            return nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1), Interpolate(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(features // 2, 32, 3, padding=1), nn.ReLU(True), nn.Conv2d(32, 3, 1), nn.Tanh())

        def mask_head():
            return nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1), nn.ReLU(True), nn.Conv2d(features // 2, 1, 1))

        self.rgb_human = rgb_head()
        self.rgb_human_object = rgb_head()
        self.mask_head = mask_head()
        self.mask_object = mask_head()


class ContactCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int=4):
        super().__init__()

        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.feedforward = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))


    def forward(self, queries, context):
        (attended, _) = self.attention(self.query_norm(queries), self.context_norm(context), self.context_norm(context), need_weights=False)
        queries = queries + attended

        return queries + self.feedforward(queries)


class AuxiliaryRelationshipEncoder(nn.Module):
    cue_channels = 6


    def __init__(self, feature_channels=256, hidden_dim=512):
        super().__init__()

        self.feature_projection = nn.Sequential(nn.Conv2d(2 * feature_channels, 192, kernel_size=1), nn.ReLU(inplace=True))
        self.cue_projection = nn.Sequential(nn.Conv2d(self.cue_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.output_projection = nn.Conv2d(256, hidden_dim, kernel_size=1)


    def forward(self, head_feature, head_object_feature, head_mask, object_mask, head_rgb, head_object_rgb, output_size):
        head_feature = F.adaptive_avg_pool2d(head_feature.detach(), output_size)
        head_object_feature = F.adaptive_avg_pool2d(head_object_feature.detach(), output_size)
        feature = self.feature_projection(torch.cat((head_feature, head_object_feature), dim=1))

        head_probability = head_mask.sigmoid().detach()
        object_probability = object_mask.sigmoid().detach()
        overlap = head_probability * object_probability

        rgb_delta = (head_object_rgb - head_rgb).abs().detach()
        cues = torch.cat((head_probability, object_probability, overlap, rgb_delta), dim=1)
        cues = F.interpolate(cues, size=output_size, mode='bilinear', align_corners=False)

        relation = self.output_projection(torch.cat((feature, self.cue_projection(cues)), dim=1))

        tokens = relation.flatten(2).transpose(1, 2)

        return torch.cat((tokens.mean(dim=1, keepdim=True), tokens), dim=1)


class ZeroInitializedRelationshipCrossAttention(nn.Module):
    def __init__(self, hidden_dim=512, attention_heads=8):
        super().__init__()

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.relation_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        self.residual = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim))

        nn.init.zeros_(self.residual[-1].weight)

        nn.init.zeros_(self.residual[-1].bias)


    def forward(self, queries, relation_tokens):
        relation = self.relation_norm(relation_tokens)
        (attended, _) = self.attention(self.query_norm(queries), relation, relation, need_weights=False)

        return queries + self.residual(attended)


class ImageSemanticAdapter(nn.Module):
    def __init__(self, width: int, strength: float, dropout: float):
        super().__init__()

        self.strength = strength
        self.norm = nn.LayerNorm(width)
        self.semantic_predictor = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Identity(), nn.Linear(width, width))
        self.film = nn.Sequential(nn.Linear(width, 2 * width), nn.Tanh())
        self.token_gate = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

        nn.init.zeros_(self.film[0].weight)

        nn.init.zeros_(self.film[0].bias)


    def forward(self, tokens):
        normalized = self.norm(tokens)

        semantic = self.semantic_predictor(normalized.mean(dim=1))
        (scale, shift) = self.film(semantic).chunk(2, dim=-1)

        gate = torch.sigmoid(self.token_gate(normalized))
        residual = gate * (normalized * scale.unsqueeze(1) + shift.unsqueeze(1))

        return (tokens + self.strength * residual, semantic)


class ContactEvidenceFusion(nn.Module):
    def __init__(self, feature_channels=256, grid_size=24):
        super().__init__()

        self.grid_size = grid_size
        self.cue_projection = nn.Sequential(nn.Conv2d(8, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU())
        self.attention = nn.Sequential(nn.Conv2d(2 * feature_channels + 96, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(), nn.Conv2d(256, 3, 1))
        self.output = nn.Sequential(nn.Conv2d(4 * feature_channels + 96, feature_channels, 1), nn.GroupNorm(16, feature_channels), nn.GELU(), nn.Conv2d(feature_channels, feature_channels, 3, padding=1))

        self.last_attention = None


    def _evidence(self, head_probability, object_probability, rgb_delta):
        size = (self.grid_size, self.grid_size)
        overlap = head_probability * object_probability
        head_uncertainty = head_probability * (1 - head_probability)
        object_uncertainty = object_probability * (1 - object_probability)

        masks = F.adaptive_max_pool2d(torch.cat((head_probability, object_probability, overlap, head_uncertainty, object_uncertainty), dim=1), size)
        rgb = F.adaptive_avg_pool2d(rgb_delta, size)

        return self.cue_projection(torch.cat((masks, rgb), dim=1))


    def _fuse(self, head_object, head, evidence):
        pair = torch.cat((head_object, head), dim=1)

        attention = F.softmax(self.attention(torch.cat((pair, evidence), dim=1)), dim=1)

        mixture = attention[:, 0:1] * head_object + attention[:, 1:2] * head
        interaction = attention[:, 2:3] * (head_object - head).abs()

        feature = self.output(torch.cat((pair, mixture, interaction, evidence), dim=1))

        return (feature, attention)


    def forward(self, head_object_feature, head_feature, head_mask, object_mask, head_rgb, head_object_rgb):
        size = (self.grid_size, self.grid_size)
        head_object = F.adaptive_avg_pool2d(head_object_feature, size)
        head = F.adaptive_avg_pool2d(head_feature, size)


        predicted_evidence = self._evidence(head_mask.sigmoid(), object_mask.sigmoid(), (head_object_rgb - head_rgb).abs())
        (predicted, attention) = self._fuse(head_object, head, predicted_evidence)

        self.last_attention = attention

        return predicted


class SemanticEvidenceFusion(nn.Module):
    def __init__(self, semantic_width, cycle=False):
        super().__init__()

        self.cycle = cycle
        self.semantic = nn.Conv2d(semantic_width, 64, 1)
        self.cue = nn.Conv2d(8, 64, 3, padding=1)
        self.gates = nn.Sequential(nn.Conv2d(256 + 256 + 128, 128, 1), nn.GELU(), nn.Conv2d(128, 3, 1))
        self.update = nn.Sequential(nn.Conv2d(256 + 256 + 128, 256, 1), nn.GroupNorm(16, 256), nn.GELU(), nn.Conv2d(256, 256, 3, padding=1))

        nn.init.normal_(self.update[-1].weight, std=0.0001)
        nn.init.zeros_(self.update[-1].bias)

        self.last_attention = None


    def forward(self, base, semantic, pair, head, hm, om, hr, pr):
        size = base.shape[-2:]
        pool = lambda x: F.adaptive_avg_pool2d(x, size)
        (h, o) = (hm.sigmoid(), om.sigmoid())
        cue = self.cue(pool(torch.cat((h, o, h * o, h * (1 - h), o * (1 - o), (pr - hr).abs()), 1)))
        semantic = self.semantic(pool(semantic))


        evidence = torch.cat((semantic, cue), 1)
        (pair, head) = (pool(pair), pool(head))

        value = base

        for _ in range(3 if self.cycle else 1):
            gates = self.gates(torch.cat((pair, head, evidence), 1)).softmax(1)


            observed = gates[:, 0:1] * pair + gates[:, 1:2] * head + gates[:, 2:3] * (pair - head).abs()
            value = value + 0.2 * self.update(torch.cat((value, observed, evidence), 1))

            if self.cycle:
                pair = pair + 0.1 * (value - base)

        self.last_attention = gates

        return value


class SpatialEvidence(nn.Module):
    def __init__(self, semantic_width, width=128):
        super().__init__()

        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(c, 32, 1), nn.GroupNorm(8, 32), nn.GELU()) for c in (semantic_width, 256, 256, 5, 9, 3)])
        self.gate = nn.Conv2d(192, 6, 1)
        self.fusion = nn.Sequential(nn.Conv2d(224, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU())
        self.position = nn.Conv2d(2, width, 1)


    def forward(self, maps):
        encoded = []

        for projection, value in zip(self.projections, maps):
            value = F.adaptive_avg_pool2d(value, (48, 48))
            item = projection(value)


            encoded.append(item)


        joined = torch.cat(encoded, 1)
        gate = self.gate(joined).softmax(1)
        mixed = sum((value * gate[:, i:i + 1] for (i, value) in enumerate(encoded)))
        feature = self.fusion(torch.cat((joined, mixed), 1))

        axis = torch.linspace(-1, 1, 48, device=feature.device, dtype=feature.dtype)
        (yy, xx) = torch.meshgrid(axis, axis, indexing='ij')

        return (feature + self.position(torch.stack((xx, yy))[None]), gate)