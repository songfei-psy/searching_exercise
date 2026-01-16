import numpy as np

EPS = 1e-9  # small constant for numerical stability


class BaseKalmanFilter:
    """Lightweight base class to share state/uncertainty bookkeeping."""

    def __init__(self, initial_state, initial_uncertainty, process_variance):
        self.state = initial_state
        self.uncertainty = initial_uncertainty
        self.process_variance = process_variance

        # History
        self.states = [initial_state]
        self.uncertainties = [initial_uncertainty]

    def _predict_common(self, motion=0.0, variance_scale=1.0):
        # State prediction (assume identity transition and unit control gain)
        self.state = self.state + motion
        # Uncertainty prediction
        self.uncertainty = self.uncertainty + self.process_variance * variance_scale
        return self.state


class SimpleKalmanFilter(BaseKalmanFilter):
    """Single-sensor 1D Kalman filter."""

    def __init__(self, initial_state, initial_uncertainty, process_variance, measurement_variance):
        super().__init__(initial_state, initial_uncertainty, process_variance)
        self.measurement_variance = measurement_variance
        self.predictions = []
        self.measurements = []

    def predict(self, motion=0.0, motion_variance_multiplier=1.0):
        state = self._predict_common(motion=motion, variance_scale=motion_variance_multiplier)
        self.predictions.append(state)
        return state

    def update(self, measurement):
        self.measurements.append(measurement)

        denom = self.uncertainty + max(self.measurement_variance, EPS)
        kalman_gain = self.uncertainty / denom

        innovation = measurement - self.state
        self.state = self.state + kalman_gain * innovation
        self.uncertainty = (1 - kalman_gain) * self.uncertainty

        self.states.append(self.state)
        self.uncertainties.append(self.uncertainty)
        return self.state, kalman_gain

    def filter(self, measurements, motions=None):
        filtered_states = []
        kalman_gains = []

        for i, z in enumerate(measurements):
            if motions is not None and i < len(motions):
                self.predict(motion=motions[i])

            state, gain = self.update(z)
            filtered_states.append(state)
            kalman_gains.append(gain)

        return np.asarray(filtered_states), np.asarray(kalman_gains)


class MultiSensorKalmanFilter(BaseKalmanFilter):
    """Two-sensor Kalman filter using information form for updates."""

    def __init__(self, initial_state, initial_uncertainty, process_variance):
        super().__init__(initial_state, initial_uncertainty, process_variance)

    def predict(self, motion=0.0, motion_model_variance=1.0):
        return self._predict_common(motion=motion, variance_scale=motion_model_variance)

    def update_with_multiple_sensors(self, measurements, measurement_variances):
        if len(measurements) != len(measurement_variances):
            raise ValueError("Measurements and variances must have the same length")

        prior_var = max(self.uncertainty, EPS)
        info_state = self.state / prior_var
        info_matrix = 1.0 / prior_var

        for z, R in zip(measurements, measurement_variances):
            var = max(R, EPS)
            info_state += z / var
            info_matrix += 1.0 / var

        if info_matrix > 0:
            self.uncertainty = 1.0 / info_matrix
            self.state = info_state * self.uncertainty
        else:
            self.uncertainty = float("inf")

        self.states.append(self.state)
        self.uncertainties.append(self.uncertainty)
        return self.state

    def batch_filter(self, time_steps, measurements_A, measurements_B, var_A, var_B):
        filtered_states = []

        for t in time_steps:
            if t > 0:
                if len(filtered_states) >= 2:
                    velocity = filtered_states[-1] - filtered_states[-2]
                else:
                    velocity = 0
                self.predict(motion=velocity)

            current_measurements = [measurements_A[t], measurements_B[t]]
            current_variances = [var_A, var_B]

            state = self.update_with_multiple_sensors(current_measurements, current_variances)
            filtered_states.append(state)

        return np.asarray(filtered_states)
