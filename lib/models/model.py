from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.backbone.siglip2 import SigLIP2
from lib.models.blocks import (
    AuxiliaryRelationshipEncoder,
    ContactCrossAttention,
    ContactEvidenceFusion,
    DPTFourHeadDecoder,
    ImageSemanticAdapter,
    LaMaFeatureInpainter,
    SemanticEvidenceFusion,
    SpatialEvidence,
    ZeroInitializedRelationshipCrossAttention,
)
from lib.models.iuv import IUVTransport


class ContactHead(nn.Module):
    def __init__(self, width=128):
        super().__init__()

        self.cues = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))

        self.update = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU())

        self.output = nn.Linear(width, 1)
        self.iuv = IUVTransport(width)
        self.iuv_norm = nn.LayerNorm(width)
        self.iuv_strength = nn.Parameter(torch.tensor(-1.4))


    def forward(self, query, image, neighbors, valid):
        local = self.iuv(image)
        query = query + self.iuv_strength.sigmoid() * self.iuv_norm(local)

        prior = query.new_zeros(query.shape[:2])
        nearby = (prior[:, neighbors] * valid[None]).sum(-1) / valid.sum(-1).clamp_min(1)
        cues = torch.stack((prior, nearby, prior - nearby, prior * (1 - prior)), -1)

        context = image.mean((2, 3))[:, None]
        state = self.update(torch.cat((query + context, self.cues(cues.to(query.dtype))), -1))

        return self.output(state).squeeze(-1)


class HECO(nn.Module):
    def __init__(self):
        super().__init__()

        self.siglip = SigLIP2()


        width, hidden = 1536, 512
        self.image_adapter = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

        self.context_projection = nn.Linear(width, hidden)
        self.skip_projections = nn.ModuleList([nn.Linear(width, hidden) for _ in range(3)])


        self.ffc = LaMaFeatureInpainter(hidden, 18)
        self.ffc_human = LaMaFeatureInpainter(hidden, 18)

        self.dpt = DPTFourHeadDecoder(hidden, features=256)
        self.dpt_human = deepcopy(self.dpt)

        for name in ('rgb_human', 'rgb_human_object', 'mask_head', 'mask_object'):
            delattr(self.dpt_human, name)

        self.decoder_context_projection = nn.Conv2d(256, hidden, 1)


        self.contact_queries = nn.Embedding(5023, hidden)
        self.vertex_position = nn.Linear(3, hidden, bias=False)
        self.cross_attention = nn.ModuleList([ContactCrossAttention(hidden, 8) for _ in range(4)])

        self.hair_hole_predictor = nn.Conv2d(hidden, 1, 1)
        self.object_hole_predictor = nn.Conv2d(hidden, 1, 1)


        self.auxiliary_relationship_encoder = AuxiliaryRelationshipEncoder(256, hidden)
        self.relationship_cross_attention = ZeroInitializedRelationshipCrossAttention(hidden, 8)


        self.semantic_adapter = ImageSemanticAdapter(width, strength=0.35, dropout=0.15)
        self.semantic_query_projection = nn.Linear(width, hidden)


        self.causal_fusion = ContactEvidenceFusion(256, 24)
        self.program_fusion = SemanticEvidenceFusion(width, cycle=True)


        self.research_evidence = SpatialEvidence(width)
        self.research_query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 128))
        self.research_geometry = nn.Sequential(nn.Linear(39, 128), nn.GELU(), nn.Linear(128, 128))
        self.research_head = ContactHead()


        self.register_buffer('flame_reference_vertices', torch.empty(5023, 3))
        self.register_buffer('mesh_neighbors', torch.empty(5023, 32, dtype=torch.long))
        self.register_buffer('mesh_neighbor_valid', torch.empty(5023, 32))
        self.register_buffer('research_coordinates', torch.empty(5023, 39))


    def _trunk(self, features, skips, dpt):
        maps = [*skips, features]
        layers = [conv(resize(x)) for x, resize, conv in zip(maps, dpt.reassemble, dpt.scratch)]
        path = dpt.refinenet[3](layers[3])

        for level in (2, 1, 0):
            if path.shape[-2:] != layers[level].shape[-2:]:
                path = F.interpolate(path, size=layers[level].shape[-2:], mode='bilinear', align_corners=True)

            path = dpt.refinenet[level](path, layers[level])

        return path


    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, 384, 384):
            raise ValueError('Expected RGB input with shape (B, 3, 384, 384)')

        batch = image.shape[0]
        vision = self.siglip(image)
        adapted, semantic = self.semantic_adapter(self.image_adapter(vision.last_hidden_state))
        semantic_grid = adapted.transpose(1, 2).reshape(batch, -1, 24, 24)
        context = self.context_projection(adapted)
        grid = context.transpose(1, 2).reshape(batch, -1, 24, 24)


        head_probability = self.hair_hole_predictor(grid).sigmoid()
        object_probability = self.object_hole_predictor(grid).sigmoid()
        union = head_probability + (1 - head_probability) * object_probability


        states = vision.hidden_states
        indices = [max(0, round((len(states) - 1) * fraction)) for fraction in (.25, .5, .75)]
        skips = [projection(states[index]).transpose(1, 2).reshape(batch, -1, 24, 24)
                 for projection, index in zip(self.skip_projections, indices)]


        pair_gate = .20 + .80 * (1 - head_probability)
        head_gate = .20 + .80 * (1 - union)

        pair = self._trunk(self.ffc(grid * pair_gate), [x * pair_gate for x in skips], self.dpt)
        head = self._trunk(self.ffc_human(grid * head_gate), [x * head_gate for x in skips], self.dpt_human)


        size = image.shape[-2:]

        def resize(value):
            return F.interpolate(value, size=size, mode='bilinear', align_corners=True)

        head_rgb = resize(self.dpt.rgb_human(head))
        pair_rgb = resize(self.dpt.rgb_human_object(pair))
        head_mask = resize(self.dpt.mask_head(head))
        object_mask = resize(self.dpt.mask_object(pair))


        decoded = self.causal_fusion(pair, head, head_mask, object_mask, head_rgb, pair_rgb)
        h, o = head_mask.sigmoid(), object_mask.sigmoid()
        maps = (
            semantic_grid, head, pair,
            torch.cat((h, o, h * o, h * (1 - h), o * (1 - o)), 1),
            torch.cat((head_rgb, pair_rgb, (pair_rgb - head_rgb).abs()), 1),
            self.causal_fusion.last_attention)

        decoded = self.program_fusion(decoded, semantic_grid, pair, head, head_mask, object_mask, head_rgb, pair_rgb)
        maps = (*maps[:-1], .5 * (maps[-1] + self.program_fusion.last_attention))


        restored = self.decoder_context_projection(F.adaptive_avg_pool2d(decoded, (24, 24)))
        fused = context + restored.flatten(2).transpose(1, 2)
        fused = torch.cat((fused.mean(dim=1, keepdim=True), fused), dim=1)

        relation = self.auxiliary_relationship_encoder(head, pair, head_mask, object_mask, head_rgb, pair_rgb, (24, 24))


        basis = (self.contact_queries.weight + self.vertex_position(self.flame_reference_vertices)).unsqueeze(0)
        basis = basis + .25 * self.semantic_query_projection(semantic).unsqueeze(1)

        queries = []

        for start in range(0, 5023, 512):
            chunk = basis[:, start:start + 512]

            for layer in self.cross_attention:
                chunk = layer(chunk, fused)

            chunk = self.relationship_cross_attention(chunk, relation)
            queries.append(chunk)


        dense = torch.cat(queries, dim=1)
        evidence, _ = self.research_evidence(maps)
        query = self.research_query(dense) + self.research_geometry(self.research_coordinates)[None]

        return self.research_head(query, evidence, self.mesh_neighbors, self.mesh_neighbor_valid)


def get_model(checkpoint, device='cuda'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    unused_prefixes = ('siglip.vision_model.head.', 'dpt.refinenet.3.resConfUnit1.', 'dpt_human.refinenet.3.resConfUnit1.')
    state = {name: value for name, value in state.items() if not name.startswith(unused_prefixes)}

    with torch.device('meta'):
        model = HECO()

    model.load_state_dict(state, strict=True, assign=True)

    model.siglip.vision_model.embeddings.position_ids = torch.arange(576).expand((1, -1))

    return model.to(device).eval()