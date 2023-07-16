import albumentations as A


def get_albumentations(CFG):
    """Returns a composite albumentation augmentation to be applied to the training images."""
    transforms = A.Compose(
        [
            #         A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.25),
            A.OneOf(
                [
                    A.RandomShadow(
                        num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=3, p=1
                    ),
                    A.ElasticTransform(alpha=120, sigma=35, alpha_affine=3, p=1.0),
                ],
                p=0.25,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomGamma(gamma_limit=(40, 450), p=1.0),
                    A.CLAHE(p=1.0),
                    A.ImageCompression(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.ColorJitter(p=1),
                    A.RGBShift(p=1),
                ],
                p=0.25,
            ),
            A.OneOf(
                [
                    A.Blur(p=1),
                    A.GaussNoise(p=1),
                ],
                p=0.1,
            ),
        ],
        p=1.0,
    )
    return None if CFG.no_aug else transforms
