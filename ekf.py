import numpy as np
import matplotlib.pyplot as plt

####Acceleration####
#Chosen a simple/random nonlinear function for convenience 
def f(x):
	return x**2

def diff_f(x):
	return x*2

true_acc = [f(x) for x in range(0, 50)]

#Random process and observation noise models
process_noise = np.random.normal(0, 100, 50)
observe_noise = np.random.normal(0, 50, 50)

#The state transition and observation models
simulated_accel_model = true_acc + process_noise
observation_model = true_acc + observe_noise

#Q: covariance of process noise
cov_process_noise = np.cov(process_noise)
#R: covariance of observation noise
cov_observe_noise = np.cov(observe_noise)
#print(cov_process_noise)
#print(cov_observe_noise)

#The update covariance and state estimates go here
covariance_estimate = [0]
state_estimate = [0]

for k in range(1,50): 
	#predict state step: x_cap_k|k-1 = f(x_cap_k-1|k-1)
	predicted_state_estimate = f(k-1)

	#calculating jacobian: F_k = diff(f(x_cap_k-1|k-1)) and H_k = diff(f(x_cap_k|k-1))
	state_transition = diff_f(k-1)
	observation_transition = diff_f(predicted_state_estimate)

	#predict covariance estimate: P_k|k-1 = F . Pk-1|k-1 . Ft + Q
	predicted_covariance_estimate = state_transition * covariance_estimate[k-1] * np.transpose(state_transition) + cov_process_noise

	#Update steps: y = z_k - h(x_cap_k-1|k-1) and S = H_k * P_k|k-1 * H_kt + Q
	measurement_residual = observation_model[k] - f(k-1)
	covariance_residual = observation_transition * predicted_covariance_estimate * np.transpose(observation_transition) + cov_observe_noise

	#K = P_k|k-1 * H_kt * S^-1
	kalman_gain = predicted_covariance_estimate * np.transpose(observation_transition) * (covariance_residual**-1)

	#x_cap_k|k = x_cap_k|k-1 + K * y and P_k|k = (I - K * H_k) * P_k|k-1
	updated_state_estimate = predicted_state_estimate + kalman_gain * measurement_residual
	updated_covariance_estimate = (1 - kalman_gain * observation_transition) * predicted_covariance_estimate

	#Appending to the final list 
	state_estimate.append(updated_state_estimate) 
	covariance_estimate.append(updated_covariance_estimate)

'''
#print(state_estimate)
plt.plot(range(len(true_acc)), true_acc, label = "True Acceleration")
plt.plot(range(len(simulated_accel_model)), simulated_accel_model, label = "Acceleration Model")
plt.plot(range(len(observation_model)), observation_model, label = "Sensor Value")
plt.plot(range(len(state_estimate)), state_estimate, label = "Estimated state")
plt.legend()
plt.show()
'''

####Velocity####
def fv(x):
	return (x**3)/3

def diff_fv(x):
	return x**2

true_vel = [fv(x) for x in range(0, 50)]

process_noise_vel = np.random.normal(0, 1500, 50)
observe_noise_vel = np.random.normal(0, 1500, 50)

simulated_vel_model = true_vel 
observation_model_vel = true_vel + observe_noise_vel

#Q
cov_process_noise_vel = np.cov(process_noise_vel)
#R
cov_observe_noise_vel = np.cov(observe_noise_vel)

covariance_estimate_vel = [0]
state_estimate_vel = []
state_estimate_vel.append(true_vel[0])

for k in range(1, 50):
	predicted_state_estimate_vel = fv(k-1)

	state_transition_vel = diff_fv(k-1)
	observation_transition_vel = diff_fv(predicted_state_estimate_vel)

	predicted_covariance_estimate_vel = state_transition_vel * covariance_estimate_vel[k-1] * np.transpose(state_transition_vel) + cov_process_noise_vel

	measurement_residual_vel = observation_model_vel[k] - fv(k-1)
	covariance_residual_vel = observation_transition_vel * predicted_covariance_estimate_vel * np.transpose(observation_transition_vel) + cov_observe_noise_vel

	kalman_gain_vel = predicted_covariance_estimate_vel * np.transpose(observation_transition_vel) * (covariance_residual_vel**-1)

	updated_state_estimate_vel = predicted_state_estimate_vel + kalman_gain_vel * measurement_residual_vel
	updated_covariance_estimate_vel = (1 - kalman_gain_vel * observation_transition_vel) * predicted_covariance_estimate_vel

	state_estimate_vel.append(updated_state_estimate_vel) 
	covariance_estimate_vel.append(updated_covariance_estimate_vel)

plt.plot(range(len(true_vel)), true_vel, label = "True Velocity")
plt.plot(range(len(simulated_vel_model)), simulated_vel_model, label = "Velocity Model")
plt.plot(range(len(observation_model_vel)), observation_model_vel, label = "Sensor Value for Velocity")
plt.plot(range(len(state_estimate_vel)), state_estimate_vel, label = "Estimated state Velocity")
plt.legend()
plt.show()

####Position####

def fp(x):
	return (x**4)/12

def diff_fp(x):
	return (x**3)/3

true_pos = [fp(x) for x in range(0, 50)]

process_noise_pos = np.random.normal(0, 10000, 50)
observe_noise_pos = np.random.normal(0, 7500, 50)

simulated_pos_model = true_pos
observation_model_pos = true_pos + observe_noise_pos
#Q
cov_process_noise_pos = np.cov(process_noise_pos)
#R
cov_observe_noise_pos = np.cov(observe_noise_pos)

covariance_estimate_pos = [0]
state_estimate_pos = []
state_estimate_pos.append(true_pos[0])

for k in range(1, 50):
	predicted_state_estimate_pos = fp(k-1)

	state_transition_pos = diff_fp(k-1)
	observation_transition_pos = diff_fp(predicted_state_estimate_vel)

	predicted_covariance_estimate_pos = state_transition_pos * covariance_estimate_pos[k-1] * np.transpose(state_transition_pos) + cov_process_noise_pos

	measurement_residual_pos = observation_model_pos[k] - fp(k-1)
	covariance_residual_pos = observation_transition_pos * predicted_covariance_estimate_pos * np.transpose(observation_transition_pos) + cov_observe_noise_pos

	kalman_gain_pos = predicted_covariance_estimate_pos * np.transpose(observation_transition_pos) * (covariance_residual_pos**-1)

	updated_state_estimate_pos = predicted_state_estimate_pos + kalman_gain_pos * measurement_residual_pos
	updated_covariance_estimate_pos = (1 - kalman_gain_pos * observation_transition_pos) * predicted_covariance_estimate_pos

	state_estimate_pos.append(updated_state_estimate_pos) 
	covariance_estimate_pos.append(updated_covariance_estimate_pos)

'''
plt.plot(range(len(true_pos)), true_pos, label = "True position")
plt.plot(range(len(simulated_pos_model)), simulated_pos_model, label = "position Model")
plt.plot(range(len(observation_model_pos)), observation_model_pos, label = "Sensor Value for position")
plt.plot(range(len(state_estimate_pos)), state_estimate_pos, label = "Estimated state position")
plt.legend()
plt.show()
'''