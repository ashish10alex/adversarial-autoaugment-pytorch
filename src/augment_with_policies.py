import pdbr
from augmentation_policies import augmentation_policies
import numpy as np

'''
Various augmentation polices
In [5]: augmentation_policies.transforms
Out[5]:
[<audiomentations.augmentations.transforms.AddGaussianNoise at 0x7ff283587f50>,
 <audiomentations.augmentations.transforms.AddShortNoises at 0x7ff283587890>,
 <audiomentations.augmentations.transforms.TimeMask at 0x7ff283587dd0>,
 <audiomentations.augmentations.transforms.FrequencyMask at 0x7ff2835f36d0>,
 <audiomentations.augmentations.transforms.TimeStretch at 0x7ff2835f3710>,
 <audiomentations.augmentations.transforms.PitchShift at 0x7ff267added0>,
 <audiomentations.core.composition.Compose at 0x7ff267addf90>]
'''

class AugmentWithPolicies(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self,audio):
        audios = [policy(audio.copy(), sample_rate=8000) for policy in self.policies] 
        return audios

augmentation_function = AugmentWithPolicies(augmentation_policies)
# audio = np.random.randn(8000)
# aug_audio = np.array(augs(audio))
# print(aug_audio)

