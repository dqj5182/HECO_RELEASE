import json
from pathlib import Path

from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel


class SigLIP2(nn.Module):
    def __init__(self):
        super().__init__()

        config = json.loads(Path(__file__).with_name('vision_config.json').read_text())
        config['vision_use_head'] = False
        self.vision_model = SiglipVisionModel(SiglipVisionConfig(**config))


    def forward(self, image):
        return self.vision_model(pixel_values=image, output_hidden_states=True)