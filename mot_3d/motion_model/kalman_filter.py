""" Many parts are borrowed from https://github.com/xinshuoweng/AB3DMOT
"""

import numpy as np
from ..data_protos import BBox
from filterpy.kalman import KalmanFilter


class KalmanFilterMotionModel:
    # 'ciim' covariance (CIIMDEV-779): measurement variances for
    # (x, y, z, yaw, l, w, h), fitted from static-object jitter in the raw
    # DCLG-4 detection stream and rounded up slightly for margin.
    # Static-object jitter measures sigma_xy at 1.4-2.5 cm, but R must also
    # absorb the centroid shifts partial occlusion causes, and a day-scale A/B
    # showed the raw fit is over-confident: a single displaced association
    # spiked the velocity state (ghost strokes to 5.2 m, 13 ID swaps vs 2 with
    # the legacy covariance). sigma_xy 5 cm + the halved acceleration PSD calm
    # the velocity gain without giving up the dims smoothing.
    CIIM_MEAS_VAR = (2.5e-3, 2.5e-3, 2.5e-3, 10.0, 3e-3, 6e-3, 6e-3)
    CIIM_VEL_VAR = 2.25       # newborn velocity prior: (1.5 m/s)^2, a brisk walk
    CIIM_VELZ_VAR = 0.25
    CIIM_ACCEL_PSD = 2.0      # white-acceleration PSD (m^2/s^3)
    CIIM_DIM_PSD = 1e-2       # box dims random-walk (cluster deformation)
    CIIM_YAW_PSD = 1e-6       # yaw carries no information (constant at source)

    def ciim_process_noise(self, dt):
        """ Continuous white-acceleration process noise for the actual time lag.

            The legacy Q = I means one square metre of process noise per predict
            call regardless of dt. This scales with the real gap: positions gain
            q*dt^4/4, velocities q*dt^2, with the q*dt^3/2 cross terms, so a long
            coast inflates the covariance the way the motion model actually
            degrades -- which is what makes the innovation matrix meaningful.
        """
        q = getattr(self, 'accel_psd', self.CIIM_ACCEL_PSD)
        Q = np.zeros((10, 10))
        for p, v in ((0, 7), (1, 8), (2, 9)):
            Q[p, p] = q * dt ** 4 / 4.0
            Q[p, v] = Q[v, p] = q * dt ** 3 / 2.0
            Q[v, v] = q * dt ** 2
        Q[3, 3] = self.CIIM_YAW_PSD * dt
        for d in (4, 5, 6):
            Q[d, d] = self.CIIM_DIM_PSD * dt
        return Q

    def __init__(self, bbox: BBox, inst_type, time_stamp, covariance='default',
                 max_speed=0.0):
        # CIIMDEV-779: max_speed (m/s, 0 = off) clamps the horizontal velocity
        # state after every update. This is a prior on the STATE, not on the
        # association: with the physical gate the filter cannot be handed a
        # super-human displacement, but on re-acquire after a coast it still
        # attributes the residual to velocity and overshoots (~10 mph seen on
        # DCLG-4). Bounding the state bounds the coast that follows
        # (max_age * max_speed) and the published speed. Measured effect on the
        # association itself: none (A/B over three hours, identical rows).
        self.max_speed = float(max_speed or 0.0)
        # the time stamp of last observation
        self.prev_time_stamp = time_stamp
        self.latest_time_stamp = time_stamp
        # define constant velocity model
        self.score = bbox.s
        self.inst_type = inst_type

        self.kf = KalmanFilter(dim_x=10, dim_z=7) 
        self.kf.x[:7] = BBox.bbox2array(bbox)[:7].reshape((7, 1))
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])
        
        self.kf.B = np.zeros((10, 1))                     # dummy control transition matrix

        # # with angular velocity
        # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
        # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
        #                       [0,1,0,0,0,0,0,0,1,0,0],
        #                       [0,0,1,0,0,0,0,0,0,1,0],
        #                       [0,0,0,1,0,0,0,0,0,0,1],  
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0],
        #                       [0,0,0,0,0,0,0,1,0,0,0],
        #                       [0,0,0,0,0,0,0,0,1,0,0],
        #                       [0,0,0,0,0,0,0,0,0,1,0],
        #                       [0,0,0,0,0,0,0,0,0,0,1]])     

        # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
        #                       [0,1,0,0,0,0,0,0,0,0,0],
        #                       [0,0,1,0,0,0,0,0,0,0,0],
        #                       [0,0,0,1,0,0,0,0,0,0,0],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0]])

        # `covariance` is either the legacy string ('default' / 'ciim') or a dict
        # {'type': 'ciim', 'sigma_xy': ..., 'vel_prior': ..., 'accel_psd': ...}
        # carrying per-scene overrides of the fitted defaults -- the dynamics of
        # the tracked agents (walking office workers vs forklifts) and the
        # effective position jitter are scene properties, the dims/z/yaw noise is
        # a detector property and stays fixed.
        cov_cfg = {}
        if isinstance(covariance, dict):
            cov_cfg = covariance
            covariance = str(cov_cfg.get('type', 'default'))
        self.covariance_type = covariance
        self.accel_psd = float(cov_cfg.get('accel_psd', self.CIIM_ACCEL_PSD))
        if covariance == 'ciim':
            # Noise model fitted on DCLG-4 person tracking (CIIMDEV-779), replacing
            # the filterpy defaults (Q = I, R = I, newborn velocity variance 10,000
            # -- i.e. every measurement assumed 1 m of noise and a newborn allowed
            # ~100 m/s). Measured on the raw detection stream, static-object
            # consecutive-frame jitter: sigma_x 0.014 m, sigma_y 0.025 m,
            # sigma_z 0.023 m; box dims wander 0.05-0.08 m (cluster deformation).
            # yaw gets a huge variance because the upstream detector emits a
            # constant yaw (-1.571 on every object), so the field carries no
            # information and must not pull the state.
            meas = list(self.CIIM_MEAS_VAR)
            if 'sigma_xy' in cov_cfg:
                meas[0] = meas[1] = float(cov_cfg['sigma_xy']) ** 2
            vel_var = self.CIIM_VEL_VAR
            if 'vel_prior' in cov_cfg:
                vel_var = float(cov_cfg['vel_prior']) ** 2
            self.kf.R = np.diag(meas)
            self.kf.P = np.diag(meas + [vel_var, vel_var, self.CIIM_VELZ_VAR])
            self.kf.Q = self.ciim_process_noise(0.1)
        else:
            # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
            self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
            self.kf.P *= 10.

            # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
            # self.kf.Q[7:, 7:] *= 0.01

        self.history = [bbox]
    
    def predict(self, time_stamp=None):
        """ For the motion prediction, use the get_prediction function.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        return

    def update(self, det_bbox: BBox, aux_info=None): 
        """ 
        Updates the state vector with observed bbox.
        """
        bbox = BBox.bbox2array(det_bbox)[:7]

        # full pipeline of kf, first predict, then update
        self.predict()

        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox[3] = new_theta

        predicted_theta = self.kf.x[3]
        if np.abs(new_theta - predicted_theta) > np.pi / 2.0 and np.abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi       
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if np.abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2

        #########################     # flip

        self.kf.update(bbox)
        self.prev_time_stamp = self.latest_time_stamp
        if self.max_speed > 0:
            speed = float(np.hypot(self.kf.x[7, 0], self.kf.x[8, 0]))
            if speed > self.max_speed:
                self.kf.x[7, 0] *= self.max_speed / speed
                self.kf.x[8, 0] *= self.max_speed / speed

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        if det_bbox.s is None:
            self.score = self.score * 0.01
        else:
            self.score = det_bbox.s
        
        cur_bbox = self.kf.x[:7].reshape(-1).tolist()
        cur_bbox = BBox.array2bbox(cur_bbox + [self.score])
        self.history[-1] = cur_bbox
        return

    def get_prediction(self, time_stamp=None):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        time_lag = time_stamp - self.prev_time_stamp
        self.latest_time_stamp = time_stamp
        self.kf.F = np.array([[1,0,0,0,0,0,0,time_lag,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,time_lag,0],
                              [0,0,1,0,0,0,0,0,0,time_lag],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])
        if self.covariance_type == 'ciim':
            # The one mutating predict() happens inside update() and reuses the F
            # and Q set here, so scale Q to the same time lag F was built with.
            self.kf.Q = self.ciim_process_noise(max(time_lag, 0.0))
        pred_x = self.kf.get_prediction()[0]
        if pred_x[3] >= np.pi: pred_x[3] -= np.pi * 2
        if pred_x[3] < -np.pi: pred_x[3] += np.pi * 2
        pred_bbox = BBox.array2bbox(pred_x[:7].reshape(-1))

        self.history.append(pred_bbox)
        return pred_bbox

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.history[-1]
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance

            P is propagated through this frame's F and Q first (both were set by
            get_prediction for the real time lag since the last update), so a
            coasting track's innovation grows the way its prediction actually
            degrades. The stale P used previously meant a Mahalanobis gate never
            widened over a coast (CIIMDEV-779).
        """
        pred_P = np.matmul(np.matmul(self.kf.F, self.kf.P), self.kf.F.T) + self.kf.Q
        return np.matmul(np.matmul(self.kf.H, pred_P), self.kf.H.T) + self.kf.R
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        return
