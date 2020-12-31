# State-Estimation

This repository contains an EKF to estimate the position, velocity and acceleration of a hyperloop pod.

A random simple non-linear function is taken as the true acceleration of the pod. Random gaussian noises are generated as the process and the observation noise.
The EKF algorithm follows, the steps and comments for the same are specified in the code. Finally, the estimated acceleration is plotted against the true acceleration and RMSE is provided to help in analysis. 
