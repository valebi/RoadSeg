from argparse import Namespace
import os

import PIL
import torch
from PIL import Image
import pickle

from roadseg.inference_diffusion import make_ensemble, make_submission
from roadseg.model.smp_models import build_model
from roadseg.utils.utils import download_file_from_google_drive, finalize, setup
from roadseg.datasets.SegmentationDatasets import OnepieceCILDataset
import numpy as np

import glob

import numpy as np

from roadseg.utils.mask_to_submission import (
    mask_to_submission_strings,
    masks_to_submission,
    save_mask_as_img,
)


def get_image_patches(image, patch_size, initial_shift_x, initial_shift_y):
    height, width = image.shape[:2]
    patch_list = []
    patch_positions = []

    for start_y in range(initial_shift_y, height, patch_size):
        for start_x in range(initial_shift_x, width, patch_size):
            end_y = start_y + patch_size
            end_x = start_x + patch_size
            if end_y <= height and end_x <= width:
                patch = image[start_y:end_y, start_x:end_x]
                patch_list.append(patch)
                patch_positions.append((start_y, start_x))

    return patch_list, patch_positions

def assemble_image(patches, patch_positions, output_shape):
    output_image = np.full(output_shape, np.nan)

    for patch, position in zip(patches, patch_positions):
        start_y, start_x = position
        end_y = start_y + patch.shape[0]
        end_x = start_x + patch.shape[1]
        output_image[start_y:end_y, start_x:end_x] = patch
    return output_image

def run_inference(imgs, CFG, model, road_class=1, fold=""):
    imgs = np.asarray(imgs).transpose([0, 3, 1, 2]).astype(np.float32)
    imgs /= 255.0

    imgs = torch.tensor(imgs, device=CFG.device, dtype=torch.float32)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(imgs), batch_size=CFG.val_batch_size, shuffle=False
    )

    pred = torch.concat([model(d) for d, in dl], axis=0)
    pred = torch.softmax(pred, dim=1).cpu().numpy()
    pred = pred[:, road_class, :, :] * 255
    pred = pred.astype(np.uint8)
    return pred

@torch.no_grad()
def generate_predictions(model, CFG, road_class=1, fold="", run_inf=True):

    model.to(CFG.device)
    model.eval()

    dirname = os.path.join(CFG.out_dir, f"fold-{fold}")
    os.makedirs(dirname, exist_ok=True)
    print("tta directory is created")

    # print(big_image_shape)
    onePieceData = OnepieceCILDataset(CFG)
    big_image_shape = onePieceData.img1.shape

    output_image = np.full(big_image_shape[:2], np.nan)

    if run_inf:
        print("starting to generate predictions")
        averagedLabels = []
        for bigImage in [onePieceData.img1, onePieceData.img2]:
            print("big Image")
            big_labels = []
            for initial_shift_x in range(0, CFG.img_size, 70):
                for initial_shift_y in range(0, CFG.img_size, 70):
                    #get the shifted patches from the test images
                    patches, positions = get_image_patches(bigImage[:,:,:3], CFG.img_size, initial_shift_x, initial_shift_y)
                    print("patches are generated")
                    print("number of patches: ", len(patches))
                    # turn it into a torch tensor and predict the outcome
                    patch_labels = run_inference(patches, CFG, model, road_class=1, fold="")
                    print("patch labels are generated")
                    big_labels.append(assemble_image(patch_labels, positions, big_image_shape[:2]))
                    print(f"assembled image and appended to big labels, shift_x : {initial_shift_x}, shift_y : {initial_shift_y}")
            big_labels_array = np.stack(big_labels)
            # take the average to get labels for the images
            averagedLabels.append(np.nanmean(big_labels_array, axis=0))
            print("averaged labels are generated")
        # save the labels
        with open(os.path.join("/home/ahmet/Documents/RoadSeg/", "averagedLabels.pkl"), "wb") as f:
            pickle.dump(averagedLabels, f)
    else:
        with open(os.path.join("/home/ahmet/Documents/RoadSeg/", "averagedLabels.pkl"), "rb") as f:
            averagedLabels = pickle.load(f)

    img_files = sorted([f for f in os.listdir(CFG.test_imgs_dir) if f.endswith(".png")])
    for index , img_file in enumerate(img_files):
        # Add 144 to get the test image labels
        print("image number: ", index)
        big_img_nr, (i, j) = onePieceData.loc_dict[index+144]
        image_label = averagedLabels[big_img_nr - 1][i:i+400, j:j+400]
        #save the image
        img = PIL.Image.fromarray(image_label.astype(np.uint8))
        img.save(os.path.join(dirname, img_file))

def print_average_labels():
    with open(os.path.join("/home/ahmet/Documents/RoadSeg/", "averagedLabels.pkl"), "rb") as f:
        averagedLabels = pickle.load(f)
    for i in range(2):
        img = PIL.Image.fromarray(averagedLabels[i].astype(np.uint8))
        img.save(os.path.join("/home/ahmet/Documents/RoadSeg/output/avLabels", f"averagedLabels{i}.png"))

def main(CFG: Namespace):
    """Main function."""
    CFG.out_dir = "/home/ahmet/Documents/RoadSeg/output"
    CFG.test_imgs_dir = "/home/ahmet/Documents/RoadSeg/data/ethz-cil-road-segmentation-2023/test/images"
    CFG.data_dir = "/home/ahmet/Documents/RoadSeg/data"
    CFG.smp_backbone = "timm-regnety_080"
    CFG.smp_model = "Unet"
    CFG.device = "cuda:0"
    CFG.train_batch_size = 32
    CFG.val_batch_size = 64


    for fold in range(5):
        CFG.initial_model = f"/home/ahmet/Documents/weightsBGHER/weights/best_epoch-finetune-fold-{fold}.bin"
        model = build_model(CFG, num_classes=2)
        generate_predictions(model, CFG, road_class=1, fold=fold, run_inf=True)

    print_average_labels()
    make_ensemble(CFG)
    #make_submission(CFG)
    image_filenames = sorted(glob.glob(f"{CFG.out_dir}/ensemble/*.png"))
    masks_to_submission(CFG.submission_file, "", *image_filenames)

if __name__ == "__main__":
    args = setup()
    try:
        main(args)
    except (Exception, KeyboardInterrupt) as e:
        print(f"Exception {e} occurred, saving a checkpoint...")
        finalize(args)
        raise e
