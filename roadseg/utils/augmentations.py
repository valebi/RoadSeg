import albumentations as A


def get_albumentations(CFG):
    """Returns a composite albumentation augmentation to be applied to the training images."""
    transforms = A.Compose(
        [
            #         A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.25),
            A.OneOf(
                [
                    # A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                ],
                p=0.15,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomGamma(p=1.0),
                    A.CLAHE(p=1.0),
                    A.JpegCompression(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.RGBShift(),
                    A.Blur(),
                    A.GaussNoise(),
                ],
                p=0.15,
            ),
            A.CoarseDropout(
                max_holes=5,
                max_height=CFG.img_size // 20,
                max_width=CFG.img_size // 20,
                min_holes=2,
                fill_value=0,
                mask_fill_value=0,
                p=0.1,
            ),
        ],
        p=1.0,
    )
    return None if CFG.no_aug else transforms
