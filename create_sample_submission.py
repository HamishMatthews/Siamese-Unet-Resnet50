import numpy as np
import pandas as pd
import h5py

from trimesh.voxel.runlength import dense_to_brle
from pathlib import Path
from collections import defaultdict

from typing import Any, Union, Dict, Literal
from numpy.typing import NDArray

# class RandomModel:
#     def __init__(self, shape):
#         self.shape = shape
#         return

#     def __call__(self, input):
#         # input is ignored, just generate some random predictions
#         return np.random.randint(0, 2, size=self.shape, dtype=bool)
    
class FixedModel:
    def __init__(self, shape) -> None:
        self.shape = shape
        return
    
    def __call__(self, input) -> Any:
        # input is ignored, just generate a mask filled with zeros, with fixed pixels set to 1
        mask = np.zeros(self.shape, dtype=bool)
        mask[100:250, 100:250] = True
        return mask

def retrieve_validation_fold(path: Union[str, Path]) -> Dict[str, NDArray]:
    result = defaultdict(dict)
    with h5py.File(path, 'r') as fp:
        for uuid, values in fp.items():
            if values.attrs['fold'] != 0:
                continue
            
            result[uuid]['post'] = values['post_fire'][...]
            # result[uuid]['pre'] = values['pre_fire'][...]

    return dict(result)

def compute_submission_mask(id: str, mask: NDArray):
    brle = dense_to_brle(mask.astype(bool).flatten())
    return {"id": id, "rle_mask": brle, "index": np.arange(len(brle))}

if __name__ == '__main__':
    validation_fold = retrieve_validation_fold('train_eval.hdf5')

    # use a list to accumulate results
    result = []
    # instantiate the model
    model = FixedModel(shape=(512, 512))
    for uuid in validation_fold:
        input_images = validation_fold[uuid]

        # perform the prediction
        predicted = model(input_images)
        # convert the prediction in RLE format
        encoded_prediction = compute_submission_mask(uuid, predicted)
        result.append(pd.DataFrame(encoded_prediction))

    # concatenate all dataframes
    submission_df = pd.concat(result)
    submission_df.to_csv('predictions.csv', index=False)