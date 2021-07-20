import pdbr
# from augmentation_policies import augmentation_policies
import numpy as np
from typing import Callable

from audiomentations import (AddGaussianNoise, AddImpulseResponse,
                             AddShortNoises, Compose, FrequencyMask,
                             PitchShift, TimeMask, TimeStretch)

augmentation_list = [AddGaussianNoise, AddShortNoises, FrequencyMask, TimeMask ]

class ComposeAugmentationPolices:
    def __init__(self, probabilities, augmentation_list=augmentation_list):
        self.probabilities = probabilities
        self.augmentation_list = augmentation_list
        self.short_noises_path = "/jmain01/home/JAD007/txk02/aaa18-txk02/Conv-TasNet/src/mini_data/train_noise"

    def __call__(self) -> Callable:
        composed_augmentations = []
        for i in range(len(self.probabilities)):
            if self.augmentation_list[i].__name__ == 'AddShortNoises':
                composed_augmentations.append(Compose([self.augmentation_list[i](self.short_noises_path, p=self.probabilities[i])]))
            else: composed_augmentations.append(Compose([self.augmentation_list[i](p=self.probabilities[i])]))
        return composed_augmentations

class AugmentWithPolicies(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self,audio):
        audios = [policy(audio.copy(), sample_rate=8000) for policy in self.policies] 
        return audios

# augmentation_policies = ComposeAugmentationPolices([0.3, 0.6, 0.1], augmentation_list=augmentation_list)
# augmentation_function = AugmentWithPolicies(augmentation_policies())
# audio = np.random.randn(8000)
# aug_audio = np.array(augmentation_function(audio))
# print(aug_audio)

