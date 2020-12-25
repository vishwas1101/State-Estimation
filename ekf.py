import numpy as np
import matplotlib.pyplot as plt

####Acceleration####
#Chosen a simple/random nonlinear function for convenience 
def f(x):
	return x**2

def diff_f(x):
	return x*2

true_acc = [f(x) for x in range(0, 50)]

process_noise = np.random.normal(0, 100, 50)
observe_noise = np.random.normal(0, 50, 50)

simulated_accel_model = true_acc + process_noise
observation_model = true_acc + observe_noise

#Q
cov_process_noise = np.cov(process_noise)
#R
cov_observe_noise = np.cov(observe_noise)
#print(cov_process_noise)
#print(cov_observe_noise)

covariance_estimate = [0]
state_estimate = [0]

for k in range(1,50): 
	#predict state step
	predicted_state_estimate = f(k-1)

	#calculating jacobian
	state_transition = diff_f(state_estimate[k-1])
	observation_transition = diff_f(predicted_state_estimate)

	#predict covariance estimate: P_k|k-1 = F . Pk-1|k-1 . Ft + Q
	predicted_covariance_estimate = state_transition * covariance_estimate[k-1] * np.transpose(state_transition) + cov_process_noise

	#Update steps:
	measurement_residual = observation_model[k] - f(predicted_state_estimate)
	covariance_residual = observation_transition * predicted_covariance_estimate * np.transpose(observation_transition) + cov_observe_noise

	kalman_gain = predicted_covariance_estimate * np.transpose(observation_transition) * (covariance_residual**-1)

	updated_state_estimate = predicted_state_estimate + kalman_gain * measurement_residual
	updated_covariance_estimate = (1 - kalman_gain * observation_transition) * predicted_covariance_estimate

	state_estimate.append(updated_state_estimate) 
	covariance_estimate.append(updated_covariance_estimate)

plt.plot(range(len(true_acc)), true_acc, label = "True Acceleration")
plt.plot(range(len(simulated_accel_model)), simulated_accel_model, label = "Acceleration Model")
plt.plot(range(len(observation_model)), observation_model, label = "Sensor Value")
plt.plot(range(len(state_estimate)), state_estimate, label = "Estimated state")
plt.legend()
plt.show()
