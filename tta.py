import itertools
from argparse import Namespace
import os
from albumentations import Resize

import PIL
import torch
from PIL import Image
import pickle

from roadseg.inference_diffusion import make_ensemble, make_submission
from roadseg.model.smp_models import build_model
from roadseg.utils.utils import download_file_from_google_drive, finalize, setup
from roadseg.datasets.SegmentationDatasets import OnepieceCILDataset
import numpy as np
import matplotlib.pyplot as plt
import glob

import numpy as np
from scipy.ndimage import rotate, zoom
from roadseg.utils.mask_to_submission import (
    mask_to_submission_strings,
    masks_to_submission,
    save_mask_as_img,
)

def get_image_patches_from_top_right(image, patch_size):
    height, width = image.shape[:2]
    patch_list = []
    patch_positions = []

    for start_y in range(0, height, patch_size):
        for start_x in range(width - patch_size, -1, -patch_size):
            end_y = start_y + patch_size
            end_x = start_x + patch_size
            if end_y <= height and end_x <= width:
                patch = image[start_y:end_y, start_x:end_x]
                patch_list.append(patch)
                patch_positions.append((start_y, start_x))

    return patch_list, patch_positions

def get_image_patches_from_bottom_left(image, patch_size):
    height, width = image.shape[:2]
    patch_list = []
    patch_positions = []

    for start_y in range(height - patch_size, -1, -patch_size):
        for start_x in range(0, width, patch_size):
            end_y = start_y + patch_size
            end_x = start_x + patch_size
            if end_y <= height and end_x <= width:
                patch = image[start_y:end_y, start_x:end_x]
                patch_list.append(patch)
                patch_positions.append((start_y, start_x))

    return patch_list, patch_positions

def get_image_patches_from_bottom_right(image, patch_size):
    height, width = image.shape[:2]
    patch_list = []
    patch_positions = []

    for start_y in range(height - patch_size, -1, -patch_size):
        for start_x in range(width - patch_size, -1, -patch_size):
            end_y = start_y + patch_size
            end_x = start_x + patch_size
            if end_y <= height and end_x <= width:
                patch = image[start_y:end_y, start_x:end_x]
                patch_list.append(patch)
                patch_positions.append((start_y, start_x))

    return patch_list, patch_positions

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
        output_y = output_shape[0]
        output_x = output_shape[1]
        patch_length = patch.shape[0]

        start_y, start_x = position
        end_y = start_y + patch_length
        end_x = start_x + patch_length

        top_border_distance = start_y
        left_border_distance = start_x
        bottom_border_distance = output_y  - end_y
        right_border_distance = output_x - end_x
        distance = min(top_border_distance, bottom_border_distance, left_border_distance, right_border_distance)

        ca_dist_from_border = int((patch_length - ca_length)/2)
        linear_cadfb = min(distance, ca_dist_from_border)
        #linear_cadfb = 0
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

def run_inference(imgs, CFG, model, road_class=1):
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

def predict_shifted(bigImage, CFG, model, shift, result_zone):
    big_image_shape = bigImage.shape
    output_image = np.full(big_image_shape[:2], np.nan)
    valid_entries = np.zeros(big_image_shape[:2])

    for initial_shift_x in range(0, CFG.img_size, shift):
        for initial_shift_y in range(0, CFG.img_size, shift):
            # get the shifted patches from the test images
            patches, positions = get_image_patches(bigImage[:, :, :3], CFG.img_size, initial_shift_x, initial_shift_y)
            print("patches are generated")
            print("number of patches: ", len(patches))
            # turn it into a torch tensor and predict the outcome
            patch_labels = run_inference(patches, CFG, model, road_class=1)
            print("patch labels are generated")

            assembled_img = assemble_image(patch_labels, positions, big_image_shape[:2], result_zone)
            output_image = np.where(np.isnan(output_image), assembled_img, output_image + np.nan_to_num(assembled_img))
            valid_entries = valid_entries + np.logical_not(np.isnan(assembled_img)).astype(int)

            print(f"assembled image and added to big labels, shift_x : {initial_shift_x}, shift_y : {initial_shift_y}")

    #since we scaled the following is to make sure there are no nan values
    patches, positions = get_image_patches_from_bottom_right(bigImage[:, :, :3], CFG.img_size)
    patch_labels = run_inference(patches, CFG, model, road_class=1)
    assembled_img = assemble_image(patch_labels, positions, big_image_shape[:2], result_zone)
    output_image = np.where(np.isnan(output_image), assembled_img, output_image + np.nan_to_num(assembled_img))
    valid_entries = valid_entries + np.logical_not(np.isnan(assembled_img)).astype(int)

    patches, positions = get_image_patches_from_bottom_left(bigImage[:, :, :3], CFG.img_size)
    patch_labels = run_inference(patches, CFG, model, road_class=1)
    assembled_img = assemble_image(patch_labels, positions, big_image_shape[:2], result_zone)
    output_image = np.where(np.isnan(output_image), assembled_img, output_image + np.nan_to_num(assembled_img))
    valid_entries = valid_entries + np.logical_not(np.isnan(assembled_img)).astype(int)

    patches, positions = get_image_patches_from_top_right(bigImage[:, :, :3], CFG.img_size)
    patch_labels = run_inference(patches, CFG, model, road_class=1)
    assembled_img = assemble_image(patch_labels, positions, big_image_shape[:2], result_zone)
    output_image = np.where(np.isnan(output_image), assembled_img, output_image + np.nan_to_num(assembled_img))
    valid_entries = valid_entries + np.logical_not(np.isnan(assembled_img)).astype(int)

    #count the number of valid entries
    nan_count = np.count_nonzero(np.isnan(output_image))
    print("nan count: ", nan_count)
    #double check
    non_nan_count = np.count_nonzero(valid_entries)
    print("non nan count double check: ", non_nan_count)
    print("output image shape: ", output_image.shape)

    output_image = output_image / valid_entries
    print("predicted image is generated")
    return output_image

def transform_back_image(img, rotation, scale, flip):
    nan_count = np.count_nonzero(np.isnan(img))
    print("nan count: ", nan_count)
    if flip == -1:
        unflipped_img = img
    else:
        unflipped_img = np.flip(img, flip)

    height = int(img.shape[0]/scale[0])
    width = int(img.shape[1]/scale[1])
    resize_transform = Resize(height, width)
    unscaled_img = resize_transform(image=unflipped_img)['image']

    unrotated_img = rotate(unscaled_img, -rotation)  # inverse of rotation is -rotation
    print("transformed back image shape: ", unrotated_img.shape)
    #plot_image(unrotated_img)
    return unrotated_img


def transform_image(img, rotation, scale, flip):
    #plot_image(img)
    rotated_img = np.zeros_like(img)
    for i in range(img.shape[2]):
        rotated_img[:, :, i] = rotate(img[:, :, i], rotation)
    #plot_image(rotated_img)
    height = int(img.shape[0]*scale[0])
    width = int(img.shape[1]*scale[1])
    resize_transform = Resize(height, width)
    scaled_img = resize_transform(image=rotated_img)['image']
    #plot_image(scaled_img)
    if flip == -1:
        flipped_img = scaled_img
    else:
        flipped_img = np.flip(scaled_img, flip)
    print("transformed image shape: ", flipped_img.shape)
    #plot_image(flipped_img)
    return flipped_img

def apply_transformations_iteratively(model, CFG, bigImage ,shift, result_zone, rotations, scales, flips):
    big_image_shape = bigImage.shape
    output_image = np.full(big_image_shape[:2], np.nan)
    valid_entries = np.zeros(big_image_shape[:2])

    for rotation in rotations:
        print("rotation: ", rotation)
        rotated_img = np.zeros_like(bigImage)
        for i in range(bigImage.shape[2]):
            rotated_img[:, :, i] = rotate(bigImage[:, :, i], rotation)
        predicted = predict_shifted(rotated_img, CFG, model, shift, result_zone)
        unrotated_img = rotate(predicted, -rotation)  # inverse of rotation is -rotation
        output_image = np.where(np.isnan(output_image), unrotated_img, output_image + np.nan_to_num(unrotated_img))
        valid_entries = valid_entries + np.logical_not(np.isnan(unrotated_img)).astype(int)

    for scale in scales:
        height = int(bigImage.shape[0] * scale[0])
        width = int(bigImage.shape[1] * scale[1])
        resize_transform = Resize(height, width)
        scaled_img = resize_transform(image=bigImage)['image']
        predicted = predict_shifted(scaled_img, CFG, model, shift, result_zone)

        height = int(bigImage.shape[0])
        width = int(bigImage.shape[1])
        resize_transform = Resize(height, width)
        scaled_back_img = resize_transform(image=predicted)['image']
        output_image = np.where(np.isnan(output_image), scaled_back_img, output_image + np.nan_to_num(scaled_back_img))
        valid_entries = valid_entries + np.logical_not(np.isnan(scaled_back_img)).astype(int)

    for flip in flips:
        if flip == -1:
            flipped_img = bigImage
        else:
            flipped_img = np.flip(bigImage, flip)
        predicted = predict_shifted(flipped_img, CFG, model, shift, result_zone)
        if flip == -1:
            flipped_back_img = predicted
        else:
            flipped_back_img = np.flip(predicted, flip)
        output_image = np.where(np.isnan(output_image), flipped_back_img, output_image + np.nan_to_num(flipped_back_img))
        valid_entries = valid_entries + np.logical_not(np.isnan(flipped_back_img)).astype(int)

    output_image = output_image / valid_entries

    return output_image

def apply_all_possible_transformations(model, CFG, bigImage ,shift, result_zone, rotations, scales, flips):
    big_image_shape = bigImage.shape
    output_image = np.full(big_image_shape[:2], np.nan)
    valid_entries = np.zeros(big_image_shape[:2])

    for (rotation, scale, flip) in itertools.product(rotations, scales, flips):
        print("rotation: ", rotation, "scale: ", scale, "flip: ", flip)
        transformed_Image = transform_image(bigImage[:, :, :3], rotation, scale, flip)
        predicted = predict_shifted(transformed_Image, CFG, model, shift, result_zone)
        assembled_img = transform_back_image(predicted, rotation, scale, flip)
        output_image = np.where(np.isnan(output_image), assembled_img, output_image + np.nan_to_num(assembled_img))
        valid_entries = valid_entries + np.logical_not(np.isnan(assembled_img)).astype(int)

    output_image = output_image / valid_entries

    return output_image


@torch.no_grad()
def generate_predictions(model, CFG, fold=""):

    model.to(CFG.device)
    model.eval()

    dirname = os.path.join(CFG.out_dir, f"fold-{fold}")
    os.makedirs(dirname, exist_ok=True)

    # print(big_image_shape)
    if os.path.isfile(os.path.join(CFG.out_dir, f"onePieceData-{CFG.img_size}.pickle")):
        with open(os.path.join(CFG.out_dir, f"onePieceData-{CFG.img_size}.pickle"), 'rb') as f:
            onePieceData = pickle.load(f)
    else:
        onePieceData = OnepieceCILDataset(CFG)
        with open(os.path.join(CFG.out_dir, f"onePieceData-{CFG.img_size}.pickle"), 'wb') as f:
            pickle.dump(onePieceData, f)


    result_zone = 350
    shift = 70#70
    rotations = [0, 90, 180, 270]
    scales = [[0.8, 0.8], [0.9, 0.9], [0.95, 0.95], [1.05, 1.05], [1.2, 1.2]] # [[0.8, 0.8, 1] , [1,1,1], [1.2, 1.2,1]]
    flips = [0 , 1]


    print(f"starting to generate predictions fold : {fold}")
    averagedLabels = []
    for bigImage in [onePieceData.img1, onePieceData.img2]:
        if CFG.tta_all_combinations:
            output_image = apply_all_possible_transformations(model, CFG, bigImage[:, :, :3], shift, result_zone, rotations, scales, flips)
        else:
            output_image = apply_transformations_iteratively(model, CFG, bigImage[:, :, :3], shift, result_zone, rotations, scales, flips)
        averagedLabels.append(output_image)
        print("averaged labels are generated")

    print_average_labels(averagedLabels, CFG)

    img_files = sorted([f for f in os.listdir(CFG.test_imgs_dir) if f.endswith(".png")])
    for index , img_file in enumerate(img_files):
        # Add 144 to get the test image labels
        print("image number: ", index)
        big_img_nr, (i, j) = onePieceData.loc_dict[index+144]
        resize_transform = Resize(400 * 12, 400 * 12)
        scaled_img = resize_transform(image=averagedLabels[big_img_nr - 1])['image']
        i = int(i* 400 / CFG.img_size)
        j = int(j * 400 / CFG.img_size)
        image_label = scaled_img[i:i+400, j:j+400]

        #save the image
        img = PIL.Image.fromarray(image_label.astype(np.uint8))
        img.save(os.path.join(dirname, img_file))

def print_average_labels(averagedLabels, CFG: Namespace):
    for i in range(2):
        img = PIL.Image.fromarray(averagedLabels[i].astype(np.uint8))
        img.save(os.path.join(CFG.out_dir, f"averagedLabels{i}.png"))

def apply_tta(CFG: Namespace):
    for fold in range(5):
        if CFG.no_finetune:
            CFG.initial_model = os.path.join(CFG.finetuned_weights_dir, f"best_epoch-finetune-fold-{fold}.bin")
        else:
            CFG.initial_model = os.path.join(CFG.log_dir,  f"weights/best_epoch-finetune-fold-{fold}.bin")
        model = build_model(CFG, num_classes=2)
        generate_predictions(model, CFG, fold=fold)

def plot_image(img):
    plt.imshow(img)
    plt.axis('off')  # Remove axes
    plt.show()

def main(CFG: Namespace):
    """Main function."""
    CFG.out_dir = "/home/ahmet/Documents/CIL Project/RoadSeg/output"
    CFG.test_imgs_dir = "/home/ahmet/Documents/CIL Project/RoadSeg/data/ethz-cil-road-segmentation-2023/test/images"
    CFG.data_dir = "/home/ahmet/Documents/CIL Project/RoadSeg/data"
    CFG.smp_backbone = "efficientnet-b7"
    CFG.smp_model = "UnetPlusPlus"
    CFG.slim = True
    CFG.decoder_depth = 5
    CFG.img_size = 416
    CFG.device = "cuda:0"
    CFG.train_batch_size = 4
    CFG.val_batch_size = 8
    CFG.experiment_name = "iterative TTA"

    for fold in range(5):
        CFG.initial_model = f"/home/ahmet/Documents/CIL Project/e7-long/weights/best_epoch-finetune-fold-{fold}.bin"
        model = build_model(CFG, num_classes=2)
        generate_predictions(model, CFG, fold=fold)

    make_ensemble(CFG)

    image_filenames = sorted(glob.glob(f"{CFG.out_dir}/ensemble/*.png"))
    masks_to_submission(CFG.submission_file, "", *image_filenames)

    #make_submission(CFG)

if __name__ == "__main__":
    args = setup()
    try:
        main(args)
    except (Exception, KeyboardInterrupt) as e:
        print(f"Exception {e} occurred, saving a checkpoint...")
        finalize(args)
        raise e
