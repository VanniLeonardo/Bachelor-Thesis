# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset
from data.archive_store import ArchiveStore, frame_members, read_manifest


SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]


class Co3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        CO3D_DIR: str = None,
        CO3D_ANNOTATION_DIR: str = None,
        storage: str = "auto",
        sequences_file: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the Co3dDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            CO3D_DIR (str): Directory path to CO3D data.
            CO3D_ANNOTATION_DIR (str): Directory path to CO3D annotations.
            sequences_file (str): Optional text file with one sequence name per line; only these
                sequences are used (relative paths are resolved against the training/ folder).
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If CO3D_DIR or CO3D_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if CO3D_DIR is None or CO3D_ANNOTATION_DIR is None:
            raise ValueError("Both CO3D_DIR and CO3D_ANNOTATION_DIR must be specified.")

        category = sorted(SEEN_CATEGORIES)

        # if self.debug:
        #     category = ["car"]

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here
        self.allowed_sequences = None
        if sequences_file is not None:
            if not osp.isabs(sequences_file):
                sequences_file = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), sequences_file)
            with open(sequences_file) as f:
                self.allowed_sequences = {line.strip() for line in f if line.strip()}
            logging.info(f"Restricting {split} split to {len(self.allowed_sequences)} sequences from {sequences_file}")


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"CO3D_DIR is {CO3D_DIR}")

        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR

        # Frames may live as loose files or inside one uncompressed zip per sequence
        # (see data/archive_store.py).  "auto" picks archives when the manifest exists.
        if storage not in ("auto", "files", "archives"):
            raise ValueError(f"storage must be auto, files or archives, got {storage!r}")
        manifest = read_manifest(CO3D_DIR) if storage in ("auto", "archives") else {}
        if storage == "archives" and not manifest:
            raise FileNotFoundError(
                f"storage='archives' but no frame manifest in {CO3D_DIR}; run scripts/repack_co3d.py"
            )
        self.storage = "archives" if manifest else "files"
        self.archive_store = ArchiveStore(CO3D_DIR) if self.storage == "archives" else None
        available = {seq: set(frames) for seq, frames in manifest.items()}
        logging.info(f"CO3D storage mode: {self.storage}")

        total_frame_num = 0
        total_sequences_before_filtering = 0
        total_sequences_after_filtering = 0

        for c in category:
            for split_name in split_name_list:
                annotation_file = osp.join(
                    self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz"
                )

                try:
                    with gzip.open(annotation_file, "r") as fin:
                        annotation = json.loads(fin.read())
                except FileNotFoundError:
                    logging.error(f"Annotation file not found: {annotation_file}")
                    continue

                for seq_name, seq_data in annotation.items():
                    total_sequences_before_filtering += 1
                    if seq_name in self.invalid_sequence:
                        continue
                    if self.allowed_sequences is not None and seq_name not in self.allowed_sequences:
                        continue

                    # Keep only the frames whose files are actually present, so a
                    # partially downloaded dataset still trains.
                    if self.storage == "archives":
                        present = available.get(seq_name)
                        if not present:
                            continue
                        valid_frames_in_sequence = [f for f in seq_data if f["filepath"] in present]
                    else:
                        valid_frames_in_sequence = []
                        for frame_data in seq_data:
                            paths = frame_members(frame_data["filepath"])
                            if all(osp.exists(osp.join(self.CO3D_DIR, p)) for p in paths.values()):
                                valid_frames_in_sequence.append(frame_data)

                    if len(valid_frames_in_sequence) < self.min_num_images:
                        continue

                    self.data_store[seq_name] = valid_frames_in_sequence
                    total_frame_num += len(valid_frames_in_sequence)
                    total_sequences_after_filtering += 1

        # Add some logging to see the effect of filtering
        logging.info(f"[{split.upper()}] Filtered dataset: Kept {total_sequences_after_filtering} out of {total_sequences_before_filtering} sequences.")
        logging.info(f"[{split.upper()}] Total valid frames available for training: {total_frame_num}")

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.allow_duplicate_img
            )

        annos = [metadata[i] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for anno in annos:
            filepath = anno["filepath"]
            members = frame_members(filepath)
            image_path = osp.join(self.CO3D_DIR, filepath)

            if self.storage == "archives":
                image = decode_image_cv2(self.archive_store.read(members["image"]))
            else:
                image = read_image_cv2(image_path)

            if self.load_depth:
                if self.storage == "archives":
                    depth_map = decode_depth(self.archive_store.read(members["depth"]), 1.0)
                    mvs_mask = decode_mask_gray(self.archive_store.read(members["depth_mask"])) > 128
                else:
                    depth_map = read_depth(osp.join(self.CO3D_DIR, members["depth"]), 1.0)
                    mvs_mask = cv2.imread(osp.join(self.CO3D_DIR, members["depth_mask"]), cv2.IMREAD_GRAYSCALE) > 128

                # The MVS mask marks depths the multi-view stereo stage considered reliable.
                depth_map[~mvs_mask] = 0

                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            extri_opencv = np.array(anno["extri"])
            intri_opencv = np.array(anno["intri"])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        set_name = "co3d"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch
