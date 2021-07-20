from audiomentations import (AddGaussianNoise, AddImpulseResponse,
                             AddShortNoises, Compose, FrequencyMask,
                             PitchShift, TimeMask, TimeStretch)

augmentation_policies = Compose(
    [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
        AddShortNoises(
            "/jmain01/home/JAD007/txk02/aaa18-txk02/Conv-TasNet/src/mini_data/train_noise",
            min_time_between_sounds=3.0,
            max_time_between_sounds=3.0,
            min_snr_in_db=0,
            max_snr_in_db=24,
            p=1,
        ),
        # TimeMask(min_band_part=0.0, max_band_part=0.50, p=1),
        # FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.10, p=1),
        # TimeStretch(0.9, 1.0, leave_length_unchanged=True, p=1),
        # PitchShift(min_semitones=1, max_semitones=2, p=1),
    ]
)
augmentation_policies_tf = Compose(
    [
        FrequencyMask(min_frequency_band=0.1, max_frequency_band=0.10, p=1),
        TimeMask(min_band_part=0.1, max_band_part=0.20, p=1, fade=True),
    ]
)
augmentation_policies.transforms.append(augmentation_policies_tf)
augmentation_policies = augmentation_policies.transforms


