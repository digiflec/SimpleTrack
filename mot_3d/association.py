import numpy as np, mot_3d.tracklet as tracklet
from . import utils
from scipy.optimize import linear_sum_assignment
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .data_protos import BBox, Validity
from .preprocessing.bbox_coarse_hash import BBoxCoarseFilter

# Cost written over a pair the per-track gate excludes, so linear_sum_assignment
# can still return a square-matrix solution but never *prefers* an excluded pair.
OUT_OF_GATE_COST = 1e6


def associate_dets_to_tracks(dets, tracks, mode, asso,
    dist_threshold=0.9, trk_innovation_matrix=None, trk_gates=None):
    """ associate the tracks with detections

        trk_gates (optional): per-track positional gate in metres. When given (only
        supported with mode 'bipartite'), a det/track pair whose xy centre distance
        exceeds the track's gate is excluded BEFORE the assignment runs, and the
        flat dist_threshold post-check is skipped. Applying the threshold only
        after the assignment lets a detection with no legitimate owner displace a
        real pairing -- the pair is then discarded, both ends go free, and the
        cascade lands on a wrong-but-in-gate match (CIIMDEV-779).
    """
    if mode == 'bipartite':
        matched_indices, dist_matrix, over_gate = \
            bipartite_matcher(dets, tracks, asso, dist_threshold,
                              trk_innovation_matrix, trk_gates)
    elif mode == 'greedy':
        matched_indices, dist_matrix = \
            greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
        over_gate = None
    unmatched_dets = list()
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    unmatched_tracks = list()
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)

    matches = list()
    for m in matched_indices:
        if over_gate is not None:
            rejected = over_gate[m[0], m[1]]
        else:
            rejected = dist_matrix[m[0], m[1]] > dist_threshold
        if rejected:
            unmatched_dets.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(2))
    return matches, np.array(unmatched_dets), np.array(unmatched_tracks)


def bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix,
                      trk_gates=None):
    if asso == 'iou':
        dist_matrix = compute_iou_distance_custom(dets, tracks, asso)
    elif asso == 'giou':
        dist_matrix = compute_iou_distance_custom(dets, tracks, asso)
    elif asso == 'm_dis':
        dist_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        dist_matrix = compute_m_distance(dets, tracks, None)

    over_gate = None
    cost_matrix = dist_matrix
    if trk_gates is not None:
        # Gate on the xy residual, not on dist_matrix: the euler cost is a 7-D
        # norm that is ~2/3 box-size jitter, so a kinematic budget applied to it
        # fragments the tracker. The full cost still RANKS the in-gate candidates.
        xy = np.empty(dist_matrix.shape)
        for d, det in enumerate(dets):
            for t, trk in enumerate(tracks):
                xy[d, t] = np.hypot(det.x - trk.x, det.y - trk.y)
        over_gate = xy > np.asarray(trk_gates)[np.newaxis, :]
        cost_matrix = np.where(over_gate, OUT_OF_GATE_COST, dist_matrix)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)
    return matched_indices, dist_matrix, over_gate


def greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    """ it's ok to use iou in bipartite
        but greedy is only for m_distance
    """
    matched_indices = list()
    
    # compute the distance matrix
    if asso == 'm_dis':
        distance_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        distance_matrix = compute_m_distance(dets, tracks, None)
    elif asso == 'iou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    num_dets, num_trks = distance_matrix.shape

    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_dets
    tracking_id_matches_to_detection_id = [-1] * num_trks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])
    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2))
    else:
        matched_indices = np.asarray(matched_indices)
    return matched_indices, distance_matrix


def compute_m_distance(dets, tracks, trk_innovation_matrix):
    """ compute l2 or mahalanobis distance
        when the input trk_innovation_matrix is None, compute L2 distance (euler)
        else compute mahalanobis distance
        return dist_matrix: numpy array [len(dets), len(tracks)]
    """
    euler_dis = (trk_innovation_matrix is None) # is use euler distance
    if not euler_dis:
        trk_inv_inn_matrices = [np.linalg.inv(m) for m in trk_innovation_matrix]
    dist_matrix = np.empty((len(dets), len(tracks)))

    for i, det in enumerate(dets):
        for j, trk in enumerate(tracks):
            if euler_dis:
                dist_matrix[i, j] = utils.m_distance(det, trk)
            else:
                dist_matrix[i, j] = utils.m_distance(det, trk, trk_inv_inn_matrices[j])
    return dist_matrix


def compute_iou_distance(dets, tracks, asso='iou'):
    iou_matrix = np.zeros((len(dets), len(tracks)))
    for d, det in enumerate(dets):
        for t, trk in enumerate(tracks):
            if asso == 'iou':
                iou_matrix[d, t] = utils.iou3d(det, trk)[1]
            elif asso == 'giou':
                iou_matrix[d, t] = utils.giou3d(det, trk)
    dist_matrix = 1 - iou_matrix
    return dist_matrix



def compute_iou_distance_custom(dets, tracks, asso='iou'):
    # Create a coarse filter for both detections and tracks
    dets_coarse_filter = BBoxCoarseFilter(grid_size=2, scaler=1)
    tracks_coarse_filter = BBoxCoarseFilter(grid_size=2, scaler=1)
    
    dets_coarse_filter.bboxes2dict(dets)
    tracks_coarse_filter.bboxes2dict(tracks)
    
    iou_matrix = np.zeros((len(dets), len(tracks)))
    
    for d, det in enumerate(dets):
        related_tracks_idxes = tracks_coarse_filter.related_bboxes(det)
        if len(related_tracks_idxes) == 0:
            continue  # Skip if no related tracks in the nearby grid
        
        for t in related_tracks_idxes:
            trk = tracks[t]
            if asso == 'iou':
                iou_matrix[d, t] = utils.iou3d(det, trk)[1]
            elif asso == 'giou':
                iou_matrix[d, t] = utils.giou3d(det, trk)
    
    dist_matrix = 1 - iou_matrix
    return dist_matrix
