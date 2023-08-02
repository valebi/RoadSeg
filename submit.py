'''
    File to run to create submission out of a folder of image segmentation map predicitons.
'''
import glob
import logging

from roadseg.utils.mask_to_submission import masks_to_submission

experiment_name = "UnetPlusPlus_efficientnet-b7_partial-fold-3__2023-07-31_00-44-51"
out_images_dir = "/Users/onat/Downloads/output/ensemble"
submission_filename = f"submission_{experiment_name}.csv"

def make_submission(out_images_dir):
    image_filenames = sorted(glob.glob(f"{out_images_dir}/*.png"))
    masks_to_submission(submission_filename, "", *image_filenames)
    try:
        import kaggle

        kaggle.api.competition_submit(
            file_name=submission_filename,
            message=f"autosubmit: {experiment_name}",
            competition="ethz-cil-road-segmentation-2023",
        )
        logging.info("Submitted output to kaggle")
    except Exception as e:
        logging.info("Failed to submit to kaggle")
        logging.info(str(e))


if __name__ == "__main__":
    make_submission(out_images_dir)