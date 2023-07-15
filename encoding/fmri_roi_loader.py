import numpy as np
import os
from typing import Tuple, Dict

def load_roi(
    resp_path: str, 
    subject_name: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(f"{resp_path}/{subject_name}", 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(f"{resp_path}/{subject_name}", 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(f"{resp_path}/{subject_name}", 'roi_masks',
            rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    # roi_names = []
    lh_roi_idx = {}
    rh_roi_idx = {}
    # lh_roi_correlation = []
    # rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                # roi_names.append(r2[1])
                lh_roi_idx[r2[1]] = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx[r2[1]] = np.where(rh_challenge_rois[r1] == r2[0])[0]
                # lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                # rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    
    return lh_roi_idx, rh_roi_idx