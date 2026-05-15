from src.augmentations.none import apply_no_augmentation
from src.augmentations.smogn_aug import apply_smogn


AUGMENTATION_RUNNERS = {
    "original": apply_no_augmentation,
    "smogn": apply_smogn,
}