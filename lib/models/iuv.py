import torch
from torch import nn
from torch.nn import functional as F


class IUVTransport(nn.Module):
    def __init__(self, width=128):
        super().__init__()

        self.charts = 19
        self.predict = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(), nn.Conv2d(width, 3 * self.charts + 1, 1))
        self.features = nn.Conv2d(width, width, 1)
        self.temperature = nn.Parameter(torch.tensor(-2.0))

        self.register_buffer('vertex_chart', torch.empty(5023, 4, dtype=torch.long))
        self.register_buffer('vertex_uv', torch.empty(5023, 4, 2))
        self.register_buffer('vertex_uv_valid', torch.empty(5023, 4, dtype=torch.bool))


    def transport_features(self, features, chart, uv):
        low_uv = F.adaptive_avg_pool2d(uv.flatten(0, 1), (24, 24)).reshape(len(features), self.charts, 2, -1)
        low_chart = F.adaptive_avg_pool2d(chart, (24, 24)).float().log_softmax(1)[:, 1:].flatten(2)
        features = F.adaptive_avg_pool2d(features, (24, 24)).flatten(2).transpose(1, 2).float()

        sigma = 0.025 + 0.2 * self.temperature.sigmoid()

        chunks = []

        for start in range(0, 5023, 256):
            c = self.vertex_chart[start:start + 256]
            target = self.vertex_uv[start:start + 256]
            valid = self.vertex_uv_valid[start:start + 256]
            predicted = low_uv[:, c].float()
            distance = (predicted - target[None, :, :, :, None]).square().sum(-2)

            logits = low_chart[:, c] - distance / (2 * sigma.square())
            logits = logits.masked_fill(~valid[None, :, :, None], -10000.0)

            correspondence = torch.logsumexp(logits, dim=2).softmax(-1)

            local = correspondence @ features
            local = local * valid.any(-1)[None, :, None]
            chunks.append(local)

        return torch.cat(chunks, 1).to(features.dtype)


    def forward(self, evidence):
        prediction = self.predict(evidence)
        chart = prediction[:, :self.charts + 1]
        uv = prediction[:, self.charts + 1:].reshape(len(evidence), self.charts, 2, *evidence.shape[-2:]).sigmoid()

        local = self.transport_features(self.features(evidence), chart, uv)

        return local.to(evidence.dtype)