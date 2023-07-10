from typing import List, Tuple

import h5py
import numpy as np


def loader(
    hdf5_file: str, folds: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    post = []
    pre = []
    masks = []
    names = []

    # Read hdf5 file and filter by fold
    with h5py.File(hdf5_file, "r") as f:
        for uuid, values in f.items():
            if "fold" in values.attrs:
                if values.attrs["fold"] not in folds:
                    continue
                if "pre_fire" not in values:
                    continue

            post.append(values["post_fire"][...])
            pre.append(values["pre_fire"][...])
            masks.append(values["mask"][...])
            names.append(uuid)

    # Convert to numpy arrays
    post = np.stack(post, axis=0).astype(np.int32)
    pre = np.stack(pre, axis=0).astype(np.int32)
    masks = np.stack(masks, axis=0).astype(np.int32)

    return post, pre, masks, names
