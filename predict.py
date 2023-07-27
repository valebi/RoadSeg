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

def assemble_image(patches, patch_positions, output_shape, ca_length):
    output_image = np.full(output_shape, np.nan)

    for patch, position in zip(patches, patch_positions):
        output_length = output_shape[0]
        patch_length = patch.shape[0]

        start_y, start_x = position
        end_y = start_y + patch_length
        end_x = start_x + patch_length

        top_border_distance = start_y
        left_border_distance = start_x
        bottom_border_distance = output_length - end_y
        right_border_distance = output_length - end_x
        distance = min(top_border_distance, bottom_border_distance, left_border_distance, right_border_distance)

        ca_dist_from_border = int((patch_length - ca_length)/2)
        linear_cadfb = min(distance, ca_dist_from_border)
#---------------------- PUT THE  IMAGE IN OUTPUT  ----------------------------
        o_y1 = start_y + linear_cadfb
        o_y2 = end_y - linear_cadfb
        o_x1 = start_x + linear_cadfb
        o_x2 = end_x - linear_cadfb

        p_y1 = linear_cadfb
        p_y2 = patch_length - linear_cadfb
        p_x1 = linear_cadfb
        p_x2 = patch_length - linear_cadfb

        #print all above variables
        #print("o_y1: ", o_y1, "o_y2: ", o_y2, "o_x1: ", o_x1, "o_x2: ", o_x2)
        #print("p_y1: ", p_y1, "p_y2: ", p_y2, "p_x1: ", p_x1, "p_x2: ", p_x2)

        output_image[o_y1: o_y2, o_x1: o_x2] = patch[p_y1: p_y2, p_x1: p_x2]

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
    if os.path.isfile("onePieceData.pickle"):
        # load onepieceData from pickle
        with open('onePieceData.pickle', 'rb') as f:
            onePieceData = pickle.load(f)
    else:
        print("The file does not exist")
        onePieceData = OnepieceCILDataset(CFG)
        # save onepieceData to pickle
        with open('onePieceData.pickle', 'wb') as f:
            pickle.dump(onePieceData, f)


    big_image_shape = onePieceData.img1.shape

    if run_inf:
        print("starting to generate predictions")
        averagedLabels = []
        for bigImage in [onePieceData.img1, onePieceData.img2]:
            print("big Image")
            output_image = np.full(big_image_shape[:2], np.nan)
            valid_entries = np.zeros(big_image_shape[:2])

            for initial_shift_x in range(0, CFG.img_size, 50):
                for initial_shift_y in range(0, CFG.img_size, 50):
                    #get the shifted patches from the test images
                    patches, positions = get_image_patches(bigImage[:,:,:3], CFG.img_size, initial_shift_x, initial_shift_y)
                    print("patches are generated")
                    print("number of patches: ", len(patches))
                    # turn it into a torch tensor and predict the outcome
                    patch_labels = run_inference(patches, CFG, model, road_class=1, fold="")
                    print("patch labels are generated")

                    assembled_img = assemble_image(patch_labels, positions, big_image_shape[:2], 300)
                    output_image = np.where(np.isnan(output_image), assembled_img, output_image + np.nan_to_num(assembled_img))
                    valid_entries = valid_entries + np.logical_not(np.isnan(assembled_img)).astype(int)

                    print(f"fold : {fold}, assembled image and added to big labels, shift_x : {initial_shift_x}, shift_y : {initial_shift_y}")
            output_image = output_image / valid_entries
            # take the average to get labels for the images
            averagedLabels.append(output_image)
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
    CFG.experiment_name = "TTA-shift=1"

    for fold in range(5):
        CFG.initial_model = f"/home/ahmet/Documents/weightsBGHER/weights/best_epoch-finetune-fold-{fold}.bin"
        model = build_model(CFG, num_classes=2)
        generate_predictions(model, CFG, road_class=1, fold=fold, run_inf=True)

    print_average_labels()
    make_ensemble(CFG)

    image_filenames = sorted(glob.glob(f"{CFG.out_dir}/ensemble/*.png"))
    masks_to_submission(CFG.submission_file, "", *image_filenames)

    make_submission(CFG)

if __name__ == "__main__":
    args = setup()
    try:
        main(args)
    except (Exception, KeyboardInterrupt) as e:
        print(f"Exception {e} occurred, saving a checkpoint...")
        finalize(args)
        raise e
