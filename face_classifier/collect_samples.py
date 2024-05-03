import os
import shutil
from tqdm import tqdm


def find_nii_files_sample(root_dir):
    nii_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if ".nii" in file and (
                "_sample_0.nii" in file
                or "_sample_1.nii" in file
                or "_sample_2.nii" in file
            ):
                nii_files.append({"file": file, "dir": root})

    return nii_files


def collect_sample_files(rootdir, output_dir):
    files = find_nii_files_sample(rootdir)

    for file_obj in tqdm(files):
        file = file_obj["file"]
        file_path = os.path.join(file_obj["dir"], file)

        # print(f"Copy from {file_path}")
        # print(f"to {os.path.join(output_dir, file)}")
        shutil.copyfile(file_path, os.path.join(output_dir, file))


if __name__ == "__main__":
    collect_sample_files(
        "/mnt/c/Users/ad-sun/Downloads/evaluation_128_fsl",
        "/mnt/c/Users/ad-sun/Downloads/fsl_128_inpainted_samples",
    )
