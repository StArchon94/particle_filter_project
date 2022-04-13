# Particle Filter Project
## Implementation Plan
* Team member

    Shengjie Lin
* How I will initialize my particle cloud

    Sample uniformly in the state space. To test it, I will draw the particles and check whether they are about uniformly distributed in the map.
* How I will update the position of the particles based on the movements of the robot

    Given motion $u_t$, for each particle with state $x_{t-1}$, sample $x_t\sim{}p(x_t|x_{t-1},u_t)$. To test it, I will compare the particles' drawing and check whether the change reflects the motion.
* How I will compute the importance weights of each particle after receiving the robot's laser scan data

    Each particle's importance weight will be proportional to $p(z_t|x_t)$. The measurement model will be either beam models or likelihood fields, as described in [Class Meeting 06](https://classes.cs.uchicago.edu/archive/2022/spring/20600-1/class_meeting_06.html). To test it, I will draw the particles with color intensity proportional to their importance weights.
* How I will normalize the particles' importance weights and resample the particles

    They are normalized so that the sum of the importance weights of all particles is 1. Then the set of particles is resampled with replacement, where the probability of a particle being sampled is its importance weight. To test it, I will draw the particles after resampling and check whether resampled particles are gathered around regions where particles previously had high importance weights.
* How I will update the estimated pose of the robot

    The estimated pose of the robot will be mean value of all particles after resampling. To test it, I will draw the robot with its estimated pose on the map.
* How I will incorporate noise into your particle filter localization

    By using a probabilistic model of $p(x_t|x_{t-1},u_t)$, such as Gaussian. To test it, I will compare the particles' drawing before and after the prediction by motion, and check whether the change reflects the motion while also showing that the particles get a bit more evenly distributed.
 * Timeline to accomplish each of the components listed above

    Use one week to accomplish `initialize_particle_cloud` and `update_particles_with_motion_model`. Use another week for `update_particle_weights_with_measurement_model`, `normalize_particles`, `resample_particles` and `update_estimated_robot_pose`.
