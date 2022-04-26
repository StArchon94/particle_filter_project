# Particle Filter Project
## Implementation Plan
* Team member

    Shengjie Lin  
    Hamlet Fernandez
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
## Objectives description
The goal in this project is to gain in-depth knowledge and experience with solving problem of robot localization using the particle filter algorithm. I will have the opportunity to
* Continue gaining experience with ROS and robot programming
* Gain a hands-on learning experience with robot localization, and specifically the particle filter algorithm
* Learn about probabilistic approaches to solving robotics problems
## High-level description
Robot localization seeks to answer the question of "where am I?" from the robot's perspective. It is formulated as a state estimation problem. To this end, I will be utilizing control data $u_t$ and measurement data $z_t$ to estimate the robot state $x_t$. One important assumption of robot localization is that the robot has access to a map of the environment, which is used for the measurement update model.

Specifically, I will use the particle filter algorithm for localization. The probabilistic distribution of the robot state is represented by a set of particles, each with its own state. For each iteration, the motion model uses $u_t$ to update each particle's state as prediction. Then the measurement model evaluates the likelihood of each particle's state against $z_t$ and the map. Finally, these likelihood values are normalized and used as weights during resampling. After these steps, we now have the updated set of particles representing the current belief of the robot state.
## GIF
![particle_filter](particle_filter.gif)
TODO: rosbag record -O filename.bag /map /scan /cmd_vel /particle_cloud /estimated_robot_pose
## Code explanation
### Initialization of particle cloud
* Code location

    Line 138 to 149.
* Code description

    Keep sampling particles uniformly within a bounding box of map elements. Only particles that are located in the free map space are kept. The process is repeated until a total of `self.num_particles` valid particles are sampled.
### Movement model
* Code location

    Line 266 to 288.
* Code description

    First I compute the difference between the last and current odometry pose. Then I applied such difference to each particle based on its current state. I used the "Rotate, Move, Rotate" option as detailed in Class Meeting 08. Noises are also applied to individual particles to better model sensor noises in the real world.
### Measurement model
* Code location

    Line 250 to 264.
* Code description

    I use the likelihood fields for range finders as described in Class Meeting 06. To reduce computational burden, the `ranges` data are subsampled by a factor of 10. Also to handle the case where a query point is outside the likelihood field, which can happen due to sensor noise and particles initially spreading across the map, I simply set the distance (to the closest obstacle) of such query points to be 1.
### Resampling
* Code location

    Line 177 to 184.
* Code description

    First I generate the indices of the particles that will be assembled into the new iteration of the particle set. This is done using `numpy.random.choice` method, where I specify the sampling weights to be the normalized importance weights of the original particle set. Then I insert deep copies of the original particles into the new particle set. Deep copying is necessary so that each particle in the new set can be modified independently.
### Incorporation of noise
* Code location

    Line 281 to 283.
* Code description

    As mentioned in "Movement model" section, I used the "Rotate, Move, Rotate" movement model. Noises are sampled individually for each of the three steps from Guassian distributions.
### Updating estimated robot pose
* Code location

    Line 239 to 248.
* Code description

    Note that this happens after the particles are resampled. I simply averaged the `x`, `y` and `yaw` of all particles. I notice that it can be tricky to average the yaws, due to the periodic nature of the angle. However, in practice I find that the particles will quickly converge so it won't become an issue.
### Optimization of parameters
* Code location

    Line 84 to 85, 93 to 95 and 121 to 134.
* Code description

    To reduce computational burden, I change the number of particles from 10000 to 3000. The threshold values for linear and angular movement before an update is performed are set as suggested. The `z` values are empirically set to model cases where a ray hits the obstacle or is working at random. `self.sigma_dist` is the standard deviation for the Gaussian noise in likelihood field model. Setting it to a larger value will account for larger noise, allowing more tolerance in the measurement update. `self.sigma_t` and `self.sigma_r` are Gaussian noise parameters for the motion prediction model. I set them to be 1/10 of the linear and angular movement threshold values before motion update. `self.delta_ang` is the subsampling rate at which `ranges` data are used. `self.default_dist` is used when the query point in `get_closest_obstacle_distance` method is out of the map.
## Challenges
There are two issues that take me quite a while to figure out. One is that the initialized particles won't show up on the map after being published. Later I realized that it takes a while before a node can publish messages after its initialization. The other issue costs me significantly more time. It turns out that when I resample the particles using `numpy.random.choice` with replacement, the returned list can contain particles referencing to the same one in the original set. Therefore, in the motion update step, modifying one particle will potentially also change some others, leading to weird behaviors. To fix this issue, I need to create deep copies of the original particles during resampling.
## Future work
The current way of handling cases where the query point in `get_closest_obstacle_distance` method is out of the map is rough. I simply use a default dist in that case. I think it desirable to treat the borders of the map as lines and compute the distance of the query point to its closest border line. Also, I did not take into account the relative pose between the laser scanner and the turtlebot. So the estimated pose is essentailly that of the laser scanner. It may be working well enough for turtlebot, but not when the robot becomes more complicated.
## Takeaways
* Using an `is_initialized` variable to indicate whether everything is set up can come in handy. Otherwise if a subscriber receives a message before things are ready, the callback function might reference variables that are yet to be defined.
* Instances of `rospy` message classes are mutable python objects. Deep copying is needed if multiple copies are to be created and treated individually.
