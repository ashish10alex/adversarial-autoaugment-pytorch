
from augmentation_policies import augmentation_policies

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


def get_augment(name):
    return augment_dict[name]

def apply_augment(audio, sub_policy):
    augment_fn = get_augment(sub_policy)
    return augment_fn(audio.copy())

class Augmentation(object):
    def __init__(self, policy):
        """
        Doc string here
        """
        self.policy = policy

    def __call__(self, audio):
        sub_policy = random.choice(self.policy)
        audio = apply_augment(audio, sub_policy)
        return audio

class MultiAugmentationAudio(object):
    def __init__(self, policies):
        self.policies = [Augmentation(policy) for policy in policies]
        print(self.policies)

    def __call__(self,audio):
        # print('function was called')
        audios = [policy(audio) for policy in self.policies] 
        return audios
