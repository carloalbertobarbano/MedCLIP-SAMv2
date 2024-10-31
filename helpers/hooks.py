"""
Author: Carlo Alberto Barbano (carlo.barbano@unito.it)
Date: 12/05/24
"""

import torch
import clip.model


class FeaturesHook:
    def __init__(self, model: clip.model.VisionTransformer, start_block=0):
        self.model = model
        self.start_block = start_block
        self.hooks = []
        self.features = []

        resblocks = model.transformer.resblocks[start_block:]
        for i, resblock in enumerate(resblocks):
            self.hooks.append(resblock.register_forward_hook(self.hook_fn))

    def hook_fn(self, module, input, output):
        self.features.append(output)

    def reset(self):
        self.features = []

    def remove(self):
        for hook in self.hooks:
            hook.remove()