import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class SimpleKalmanFilter:
    """最简单的一维卡尔曼滤波器"""
    def __init__(self, prior_state, prior_uncertainty, process_variance, measurement_variance):
        """
        参数:
        prior_state: 初始状态估计
        prior_uncertainty: 初始不确定性（协方差）
        process_variance: 过程噪声方差（Q）
        measurement_variance: 观测噪声方差（R）
        """
        self.state = prior_state
        self.uncertainty = prior_uncertainty
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # 历史记录
        self.states = [prior_state]
        self.uncertainties = [prior_uncertainty]
        self.predictions = []
        self.measurements = []
        
    def predict(self, motion=0, motion_variance_multiplier=1):
        """预测步：状态向前传播"""
        # 状态预测：x = Fx + Bu，这里假设F=1, B=1
        self.state = self.state + motion
        
        # 不确定性预测：P = FPF^T + Q
        self.uncertainty = self.uncertainty + self.process_variance * motion_variance_multiplier
        
        self.predictions.append(self.state)
        return self.state
        
    def update(self, measurement):
        """更新步：用观测修正状态"""
        # 保存测量值
        self.measurements.append(measurement)
        
        # 计算卡尔曼增益：K = P / (P + R)
        kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_variance)
        
        # 状态更新：x = x + K*(z - x)
        innovation = measurement - self.state
        self.state = self.state + kalman_gain * innovation
        
        # 不确定性更新：P = (1 - K) * P
        self.uncertainty = (1 - kalman_gain) * self.uncertainty
        
        # 保存历史
        self.states.append(self.state)
        self.uncertainties.append(self.uncertainty)
        
        return self.state, kalman_gain
    
    def filter(self, measurements, motions=None):
        """批量滤波"""
        filtered_states = []
        kalman_gains = []
        
        for i, z in enumerate(measurements):
            # 如果有运动命令，先预测
            if motions is not None and i < len(motions):
                self.predict(motion=motions[i])
            
            # 更新
            state, gain = self.update(z)
            filtered_states.append(state)
            kalman_gains.append(gain)
            
        return filtered_states, kalman_gains
    
    
class MultiSensorKalmanFilter:
    """多传感器卡尔曼滤波器"""
    def __init__(self, prior_state, prior_uncertainty, process_variance):
        """
        参数:
        prior_state: 初始状态估计
        prior_uncertainty: 初始不确定性
        process_variance: 过程噪声方差
        """
        self.state = prior_state
        self.uncertainty = prior_uncertainty
        self.process_variance = process_variance
        
        # 历史记录
        self.states = [prior_state]
        self.uncertainties = [prior_uncertainty]
        
    def predict(self, motion=0, motion_model_variance=1.0):
        """预测步"""
        # 状态预测
        self.state = self.state + motion
        
        # 不确定性预测
        self.uncertainty = self.uncertainty + self.process_variance * motion_model_variance
        
        return self.state
        
    def update_with_multiple_sensors(self, measurements, measurement_variances):
        """
        用多个传感器同时更新
        
        参数:
        measurements: 传感器测量值列表
        measurement_variances: 对应传感器的噪声方差列表
        """
        if len(measurements) != len(measurement_variances):
            raise ValueError("测量值和方差列表长度必须相同")
            
        n_sensors = len(measurements)
        
        # 多传感器更新：信息形式
        # 先验信息
        info_state = self.state / self.uncertainty if self.uncertainty > 0 else 0
        info_matrix = 1.0 / self.uncertainty if self.uncertainty > 0 else 0
        
        # 添加每个传感器的信息
        for z, R in zip(measurements, measurement_variances):
            info_state += z / R
            info_matrix += 1.0 / R
        
        # 转换回状态形式
        if info_matrix > 0:
            self.uncertainty = 1.0 / info_matrix
            self.state = info_state * self.uncertainty
        else:
            # 防止除零
            self.uncertainty = float('inf')
            
        # 保存历史
        self.states.append(self.state)
        self.uncertainties.append(self.uncertainty)
        
        return self.state
    
    def batch_filter(self, time_steps, measurements_A, measurements_B, var_A, var_B):
        """批量处理两个传感器的数据"""
        filtered_states = []
        
        for t in time_steps:
            # 预测：假设我们知道运动模型是抛物线，但这里我们不知道精确模型
            # 所以使用简单的速度估计作为预测
            if t > 0:
                # 简单预测：基于前两个点的斜率
                if len(filtered_states) >= 2:
                    velocity = filtered_states[-1] - filtered_states[-2]
                else:
                    velocity = 0
                self.predict(motion=velocity)
            
            # 用两个传感器同时更新
            current_measurements = [measurements_A[t], measurements_B[t]]
            current_variances = [var_A, var_B]
            
            state = self.update_with_multiple_sensors(current_measurements, current_variances)
            filtered_states.append(state)
            
        return filtered_states
    

class ParticleFilter:
    """简单的粒子滤波器实现"""

    def __init__(self, num_particles, state_dim, motion_model, obs_model, process_noise, observation_noise):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.motion_model = motion_model
        self.obs_model = obs_model
        self.process_noise = process_noise
        self.observation_noise = observation_noise

        # 初始化粒子和权重(均匀分布)
        self.particles = np.random.randn(self.num_particles, self.state_dim)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(self):
        """预测步骤，对粒子应用运动模型并添加过程噪声"""
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), self.process_noise, self.num_particles)
        for i in range(self.num_particles):
            self.particles[i] = self.motion_model(self.particles[i]) + noise[i]

    def update(self, measurement):
        inv_R = np.linalg.inv(self.observation_noise)
        for i in range(self.num_particles):
            predicted_obs = self.obs_model(self.particles[i])
            error = measurement - predicted_obs
            likelihood = np.exp(-0.5 * error.T @ inv_R @ error)  # 高斯似然，可以改为其他分布
            self.weights[i] = likelihood
        self.weights += 1e-300  # 防止全为零
        self.weights /= np.sum(self.weights)

    def effective_sample_size(self):
        """返回有效样本量，用于评估退化程度"""
        return 1.0 / np.sum(np.square(self.weights))

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)  # 重采样，有放回
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

