import argparse
import json
import glob
import os
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="../waymo_split/official_train_w_dynamic_w_ego_motion_gt_30m_good_voxel.json")
    parser.add_argument("--input_dir", type=str, default="../waymo_webdataset")
    parser.add_argument("--output_dir", type=str, default="/dev/shm/waymo_webdataset")
    args = parser.parse_args()

    with open(args.split, "rb") as f:
        splits = json.load(f)

    split_set = set()

    for s in splits:
        split_set.add(s)

    files = []
    for filename in glob.glob(os.path.join(args.input_dir, '**/*.tar'), recursive=True):
        relative_filename = os.path.relpath(filename, args.input_dir)
        if relative_filename.split('/')[-1].split('.')[0] in split_set:
            files.append(relative_filename)

    for file in files:
        output_path = os.path.join(args.output_dir, file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(os.path.join(args.input_dir, file), output_path)