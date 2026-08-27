from copy import deepcopy
import numpy as np, mot_3d.tracklet as tracklet, mot_3d.utils as utils
from .redundancy import RedundancyModule
from scipy.optimize import linear_sum_assignment
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .data_protos import BBox, Validity
from .association import associate_dets_to_tracks
from . import visualization
from mot_3d import redundancy
import pdb, os
import time
import logging
class MOTModel:
    def __init__(self, configs):
        self.trackers = list()         # tracker for each single tracklet
        self.frame_count = 0           # record for the frames
        self.count = 0                 # record the obj number to assign ids
        self.time_stamp = None         # the previous time stamp
        self.redundancy = RedundancyModule(configs) # module for no detection cases

        non_key_redundancy_config = deepcopy(configs)
        non_key_redundancy_config['redundancy'] = {
            'mode': 'mm',
            'det_score_threshold': {'giou': 0.1, 'iou': 0.1, 'euler': 0.1},
            'det_dist_threshold': {'giou': -0.5, 'iou': 0.1, 'euler': 4}
        }
        self.non_key_redundancy = RedundancyModule(non_key_redundancy_config)

        self.configs = configs
        self.match_type = configs['running']['match_type']
        self.score_threshold = configs['running']['score_threshold']
        self.asso = configs['running']['asso']
        self.asso_thres = configs['running']['asso_thres'][self.asso]
        self.motion_model = configs['running']['motion_model']

        self.max_age = configs['running']['max_age_since_update']
        self.min_hits = configs['running']['min_hits_to_birth']

        # CIIMDEV-779: optional per-track kinematic association gate, dict with
        # keys floor / speed_factor / accel (see kinematic_gates below). None
        # keeps the legacy behaviour: flat asso_thres applied after assignment.
        self.kinematic_gate = configs['running'].get('kinematic_gate', None)
        # CIIMDEV-779 ghost suppression (see suppress_ghosts); 0 disables.
        self.ghost_dist = float(configs['running'].get('ghost_suppress_dist', 0.0) or 0.0)
        self.ghost_frames = int(configs['running'].get('ghost_suppress_frames', 3))
        # CIIMDEV-779: person prior on the detection footprint (0 = off). A box
        # wider than this in either horizontal dimension is not one person (the
        # dbt branch re-segments a group or a desk around a live track into a
        # 2 m "ADULT" the classifier would never accept) and can neither
        # continue nor birth a track.
        self.max_det_footprint = float(configs['running'].get('max_det_footprint', 0.0) or 0.0)

    def person_sized(self, det):
        if self.max_det_footprint <= 0:
            return True
        return max(float(det.l), float(det.w)) <= self.max_det_footprint

    def kinematic_gates(self, time_stamp):
        """ Per-track positional association gate in metres.

            The prediction sits at +v*dt; a target that reverses at constant speed
            lands at -v*dt, so the residual is bounded by speed_factor(=2)*v*dt.
            The accel term is the unmodelled acceleration: an unknown a over dt
            displaces the target by a*dt^2/2 whatever its current speed, which is
            what re-opens the gate for a track that was STATIONARY when it was
            lost (v~0 keeps the velocity term at zero however long the coast).
            dt only advances while unobserved, so the gate widens exactly as far
            as a coast has degraded the estimate, up to the flat asso_thres cap.
        """
        cfg = self.kinematic_gate
        gates = list()
        for trk in self.trackers:
            gate = self.asso_thres
            try:
                kf = trk.motion_model.kf
                speed = np.hypot(float(kf.x[7, 0]), float(kf.x[8, 0]))
                dt = max(time_stamp - trk.motion_model.prev_time_stamp, 0.0)
                if cfg.get('max_speed', 0.0) > 0:
                    # Physical gate (CIIMDEV-779 round 3): the target is within
                    # floor + v_max * (time since last observed) of where it was
                    # LAST OBSERVED, whatever the filter believes its velocity
                    # is. Centred there too (see gate_centres), so a poisoned
                    # velocity can neither move the gate nor widen it, and no
                    # association can ever imply a super-human displacement.
                    gate = min(gate, cfg['floor'] + cfg['max_speed'] * dt)
                else:
                    gate = min(gate, cfg['floor'] + cfg['speed_factor'] * speed * dt
                               + 0.5 * cfg['accel'] * dt * dt)
            except AttributeError:
                pass    # non-KF motion model: fall back to the flat cap
            gates.append(gate)
        return gates

    def gate_centres(self, trk_preds):
        """ Where each track's gate is centred: the last OBSERVED position under
            the physical gate (the KF state only moves on an update, so x[0:2] is
            exactly that), the prediction otherwise.
        """
        cfg = self.kinematic_gate
        centres = list()
        for trk, pred in zip(self.trackers, trk_preds):
            if cfg is not None and cfg.get('max_speed', 0.0) > 0:
                try:
                    kf = trk.motion_model.kf
                    centres.append((float(kf.x[0, 0]), float(kf.x[1, 0])))
                    continue
                except AttributeError:
                    pass
            centres.append((pred.x, pred.y))
        return centres

    def second_stage(self, input_data, matched, unmatched_dets, unmatched_trks):
        """ CIIMDEV-779: continue unmatched tracks on the detections stage 1 left.

            redundancy mode 'kinematic'. Candidates are every detection stage 1 did
            not consume and that scores above the redundancy score floor: the
            sub-threshold ones it never saw (a person whose classifier score
            flickered under score_threshold for a frame -- 44% of DCLG-4's coasted
            rows had one within 0.5 m) and the above-threshold ones it left
            unmatched (which would otherwise birth a duplicate track next to the
            coasting one). Same per-track kinematic gate as stage 1, applied on
            the xy residual, one bipartite assignment on that residual.

            Unlike the stock 'mm' redundancy, a match here hands the REAL
            detection to the track (the KF corrects to it, the published score is
            the detection's), so the row goes out as an observation. Stage 1 keeps
            priority: only its leftovers on both sides take part.

            Returns ({track index: det index}, unmatched_dets minus the consumed).
        """
        dets = input_data.dets
        min_s = self.redundancy.det_score
        leftover = set(int(i) for i in unmatched_dets)
        cand = [i for i, det in enumerate(dets)
                if det.s > min_s and (det.s < self.score_threshold or i in leftover)
                and self.person_sized(det)]
        trks = [int(t) for t in unmatched_trks]
        if len(cand) == 0 or len(trks) == 0:
            return {}, unmatched_dets
        gates = self._trk_gates
        if gates is None:
            gates = [self.asso_thres] * len(self.trackers)
        # A continuation is conservative: the detection has to be near where the
        # track IS, so the stage-1 gate (which widens with the coasted speed --
        # exactly the poisoned speed a runaway carries) is capped by a flat
        # radius, redundancy det_dist_threshold when it is positive.
        cap = self.redundancy.det_threshold if self.redundancy.det_threshold > 0 else np.inf
        # Neighbour ownership: a detection that sits closer to a track stage 1
        # already matched (with its own detection) is a fragment of that
        # neighbour, not this track's continuation.
        owned = [self._trk_preds[int(m[1])] for m in matched]
        centres = self._trk_centres
        if centres is None:
            centres = [(p.x, p.y) for p in self._trk_preds]
        cost = np.empty((len(cand), len(trks)))
        for a, i in enumerate(cand):
            d_owned = min([np.hypot(dets[i].x - p.x, dets[i].y - p.y) for p in owned] or [np.inf])
            for b, t in enumerate(trks):
                cx, cy = centres[t]
                xy = np.hypot(dets[i].x - cx, dets[i].y - cy)
                ok = xy <= min(gates[t], cap) and xy <= d_owned
                cost[a, b] = xy if ok else 1e6
        row_ind, col_ind = linear_sum_assignment(cost)
        second = {}
        for a, b in zip(row_ind, col_ind):
            if cost[a, b] < 1e6:
                second[trks[b]] = cand[a]
        consumed = set(second.values())
        unmatched_dets = np.array([i for i in unmatched_dets if int(i) not in consumed])
        return second, unmatched_dets

    def suppress_ghosts(self):
        """ CIIMDEV-779: retire a coasting track that is sitting on an observed one.

            After the frame's updates, a track that was NOT associated this frame
            (life recent_state 0) and whose state lies within ghost_dist of a track
            that WAS observed this frame is a duplicate of that track -- the
            person is being tracked by the neighbour; this one is a ghost that
            would otherwise drift on its frozen velocity until max_age. After
            ghost_frames consecutive such frames it is marked dead. The counter
            resets whenever the track is observed or the neighbour moves off.
        """
        observed = [trk for trk in self.trackers if trk.life_manager.recent_state != 0]
        for trk in self.trackers:
            if trk.life_manager.recent_state != 0:
                trk.ghost_count = 0
                continue
            st = trk.get_state()
            on_top = any(np.hypot(st.x - o.get_state().x, st.y - o.get_state().y) <= self.ghost_dist
                         for o in observed)
            trk.ghost_count = getattr(trk, 'ghost_count', 0) + 1 if on_top else 0
            if trk.ghost_count >= self.ghost_frames:
                trk.life_manager.state = 'dead'

    @property
    def has_velo(self):
        return not (self.motion_model == 'kf' or self.motion_model == 'fbkf' or self.motion_model == 'ma')
    
    def frame_mot(self, input_data: FrameData):
        """ For each frame input, generate the latest mot results
        Args:
            input_data (FrameData): input data, including detection bboxes and ego information
        Returns:
            tracks on this frame: [(bbox0, id0), (bbox1, id1), ...]
        """
        self.frame_count += 1

        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp

        if not input_data.aux_info['is_key_frame']:
            result = self.non_key_frame_mot(input_data)
            return result
        start_time = time.perf_counter()
        if 'kf' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.forward_step_trk(input_data)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.debug(f"forward_step_trk took {duration:.6f} seconds")
        
        start_time = time.perf_counter()
        time_lag = input_data.time_stamp - self.time_stamp
        # CIIMDEV-779: second-stage continuation of the tracks stage 1 left
        # unmatched, on the detections it did not consume (see second_stage).
        second = {}
        if self.redundancy.mode == 'kinematic' and self.match_type == 'bipartite':
            second, unmatched_dets = self.second_stage(input_data, matched, unmatched_dets, unmatched_trks)
        # update the matched tracks
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]),
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego,
                    frame_index=self.frame_count, pc=input_data.pc,
                    dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            elif t in second:
                # Continue the track on the real detection: the KF is corrected
                # to it and the published score is the detection's own, so a
                # consumer sees an observation, not a prediction.
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=3, bbox=input_data.dets[second[t]],
                    ego=input_data.ego, frame_index=self.frame_count,
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            else:
                result_bbox, update_mode, aux_info = self.redundancy.infer(trk, input_data, time_lag)
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_mode, bbox=result_bbox, 
                    ego=input_data.ego, frame_index=self.frame_count, 
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
        
        # create new tracks for unmatched detections
        for index in unmatched_dets:
            if self.has_velo:
                aux_info = {
                    'velo': list(input_data.aux_info['velos'][index]), 
                    'is_key_frame': input_data.aux_info['is_key_frame']}
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}

            track = tracklet.Tracklet(self.configs, self.count, input_data.dets[index], input_data.det_types[index], 
                self.frame_count, aux_info=aux_info, time_stamp=input_data.time_stamp)
            self.trackers.append(track)
            self.count += 1
        
        if self.ghost_dist > 0:
            self.suppress_ghosts()

        # remove dead tracks
        track_num = len(self.trackers)
        for index, trk in enumerate(reversed(self.trackers)):
            if trk.death(self.frame_count):
                self.trackers.pop(track_num - 1 - index)

        # output the results
        result = list()
        for trk in self.trackers:
            state_string = trk.state_string(self.frame_count)
            state = trk.get_state()
            # CIIMDEV-779: expose the filter's own velocity on the output bbox so
            # a consumer can report it instead of finite-differencing poses
            try:
                kf = trk.motion_model.kf
                state.vx, state.vy = float(kf.x[7, 0]), float(kf.x[8, 0])
            except AttributeError:
                pass
            result.append((state, trk.id, state_string, trk.det_type))

        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.debug(f"After forward_step_trk took {duration:.6f} seconds")
        return result
    
    def forward_step_trk(self, input_data: FrameData):
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets)
                       if det.s >= self.score_threshold and self.person_sized(det)]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        start_time = time.perf_counter()
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.debug(f"Prediction took {duration:.6f} seconds")
        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 
        trk_gates = None
        if self.kinematic_gate is not None and self.match_type == 'bipartite':
            trk_gates = self.kinematic_gates(input_data.time_stamp)
        # kept for second_stage: predict() has side effects, so it must not
        # be called a second time for the same frame
        self._trk_preds = trk_preds
        self._trk_gates = trk_gates
        self._trk_centres = self.gate_centres(trk_preds) if trk_gates is not None else None
        start_time = time.perf_counter()
        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds,
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix,
            trk_gates=trk_gates, trk_centres=self._trk_centres)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.debug(f"Association took {duration:.6f} seconds")
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks
    
    def non_key_forward_step_trk(self, input_data: FrameData):
        """ tracking on non-key frames (for nuScenes)
        """
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= 0.5]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        start_time = time.perf_counter()
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.debug(f"Non Keyframe prediction took {duration:.6f} seconds")
        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers] 

        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            self.match_type, self.asso, self.asso_thres, trk_innovation_matrix)
        
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks
    
    def non_key_frame_mot(self, input_data: FrameData):
        """ tracking on non-key frames (for nuScenes)
        """
        self.frame_count += 1
        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp
        start_time = time.perf_counter()
        if 'kf' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.non_key_forward_step_trk(input_data)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.debug(f"non_key_forward_step_trk took {duration:.6f} seconds")
        
        start_time = time.perf_counter()
        time_lag = input_data.time_stamp - self.time_stamp

        redundancy_bboxes, update_modes = self.non_key_redundancy.bipartite_infer(input_data, self.trackers)
        # update the matched tracks
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]), 
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego, 
                    frame_index=self.frame_count, pc=input_data.pc, 
                    dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_modes[t], bbox=redundancy_bboxes[t], 
                    ego=input_data.ego, frame_index=self.frame_count, 
                    pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
        
        # output the results
        result = list()
        for trk in self.trackers:
            state_string = trk.state_string(self.frame_count)
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))

        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logging.debug(f"after non_key_forward_step_trk took {duration:.6f} seconds")
        return result