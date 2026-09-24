# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from hydra import compose, initialize
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate the model with a Hydra config; extra arguments are Hydra overrides "
        "(e.g. `mode=val checkpoint.resume_checkpoint_path=/path/to/ckpt.pt limit_val_batches=1000`)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of the config file in training/config (without .yaml extension, default: default)",
    )
    args, overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
