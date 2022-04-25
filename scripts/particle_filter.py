#!/usr/bin/env python3

import math
from copy import deepcopy

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from tf import TransformBroadcaster, TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from likelihood_field import LikelihoodField


def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

    ang = euler_from_quaternion([
        p.orientation.x,
        p.orientation.y,
        p.orientation.z,
        p.orientation.w])[2]
    if ang > np.pi:
        ang -= np.pi * 2
    return ang


def compute_prob_zero_centered_gaussian(dist, sd):
    """ Takes in distance from zero (dist) and standard deviation (sd) for gaussian
        and returns probability (likelihood) of observation """
    c = 1.0 / (sd * math.sqrt(2 * math.pi))
    prob = c * math.exp((-math.pow(dist, 2)) / (2 * math.pow(sd, 2)))
    return prob


class Particle:
    def __init__(self, pose, w=1):
        # particle pose (Pose object from geometry_msgs)
        self.pose = pose

        # particle weight
        self.w = w


class Map:
    def __init__(self, map) -> None:
        h = map.info.height
        w = map.info.width
        self.map = np.reshape(map.data, (h, w))
        self.res = map.info.resolution
        o = map.info.origin.position
        self.x0 = o.x
        self.y0 = o.y

    def query(self, x, y):
        r = round((y - self.y0) / self.res)
        c = round((x - self.x0) / self.res)
        return self.map[r, c]


class ParticleFilter:
    def __init__(self):
        # once everything is setup, initialized will be set to true
        self.initialized = False

        # initialize this particle filter node
        rospy.init_node('turtlebot3_particle_filter')

        # set the topic names and frame names
        self.base_frame = "base_footprint"
        self.map_topic = "map"
        self.odom_frame = "odom"
        self.scan_topic = "scan"

        # initialize the likelihood field used in the measurement update
        self.likelihood_field = LikelihoodField()
        # get left, right, bottom, top boundaries of obstacles
        (self.l, self.r), (self.b, self.t) = self.likelihood_field.get_obstacle_bounding_box()
        # get the map
        self.map = Map(self.likelihood_field.map)

        # the number of particles used in the particle filter
        self.num_particles = 10000

        # initialize the particle cloud array
        self.particle_cloud = []

        # initialize the estimated robot pose
        self.robot_estimate = Pose()

        # set threshold values for linear and angular movement before we preform an update
        self.lin_mvmt_threshold = 0.2
        self.ang_mvmt_threshold = np.pi / 6

        # last odom pose
        self.odom_pose_last_motion_update = None

        # Setup publishers and subscribers

        # publish the current particle cloud
        self.particles_pub = rospy.Publisher("particle_cloud", PoseArray, queue_size=1)

        # publish the estimated robot pose
        self.robot_estimate_pub = rospy.Publisher("estimated_robot_pose", PoseStamped, queue_size=1)

        # subscribe to the lidar scan from the robot
        rospy.Subscriber(self.scan_topic, LaserScan, self.robot_scan_received, queue_size=1)

        # enable listening for and broadcasting corodinate transforms
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

        # wait 1 sec before publishing
        rospy.sleep(1)

        # intialize the particle cloud
        self.initialize_particle_cloud()

        # parameters for the measurement update model
        self.z_hit = 0.8
        self.z_random = 0.1
        self.sigma_dist = 0.1

        # noise parameters for the motion prediction model
        self.sigma_t = self.lin_mvmt_threshold * 0.1
        self.sigma_r = self.ang_mvmt_threshold * 0.1

        # angular interval (in degree) at which the ranges data is used
        self.delta_ang = 10

        # default distance used when the query point is out of the likelihood field map
        self.default_dist = 1

        self.initialized = True

    def initialize_particle_cloud(self):
        i = 0
        while i < self.num_particles:
            # sample uniformly within the bounding box
            x = np.random.uniform(self.l, self.r)
            y = np.random.uniform(self.b, self.t)
            # check if the particle's position is free space in the map
            if not self.map.query(x, y):
                ang = np.random.uniform(-np.pi, np.pi)
                p = Particle(Pose(Point(x, y, 0), Quaternion(*quaternion_from_euler(0, 0, ang))))
                self.particle_cloud.append(p)
                i += 1

        self.normalize_particles()
        self.publish_particle_cloud()
        self.update_estimated_robot_pose()
        self.publish_estimated_robot_pose()

    def normalize_particles(self):
        # make all the particle weights sum to 1.0
        w_sum = sum([p.w for p in self.particle_cloud])
        for p in self.particle_cloud:
            p.w /= w_sum

    def publish_particle_cloud(self):
        particle_cloud_pose_array = PoseArray()
        particle_cloud_pose_array.header = Header(stamp=rospy.Time.now(), frame_id=self.map_topic)

        for part in self.particle_cloud:
            particle_cloud_pose_array.poses.append(part.pose)

        self.particles_pub.publish(particle_cloud_pose_array)

    def publish_estimated_robot_pose(self):
        robot_pose_estimate_stamped = PoseStamped()
        robot_pose_estimate_stamped.pose = self.robot_estimate
        robot_pose_estimate_stamped.header = Header(stamp=rospy.Time.now(), frame_id=self.map_topic)
        self.robot_estimate_pub.publish(robot_pose_estimate_stamped)

    def resample_particles(self):
        # generate indices of choice according to the particle weights
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=[p.w for p in self.particle_cloud])
        particle_cloud = []
        for i in indices:
            # use deepcopy so that particles won't share memory
            particle_cloud.append(deepcopy(self.particle_cloud[i]))
        self.particle_cloud = particle_cloud

    def robot_scan_received(self, data):
        # wait until initialization is complete
        if not self.initialized:
            return

        # we need to be able to transfrom the laser frame to the base frame
        if not self.tf_listener.canTransform(self.base_frame, data.header.frame_id, data.header.stamp):
            return

        # wait for a little bit for the transform to become avaliable (in case the scan arrives
        # a little bit before the odom to base_footprint transform was updated)
        self.tf_listener.waitForTransform(self.base_frame, self.odom_frame, data.header.stamp, rospy.Duration(0.5))

        # calculate the pose of the laser distance sensor
        # p = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id=data.header.frame_id))
        # self.laser_pose = self.tf_listener.transformPose(self.base_frame, p)

        # determine where the robot thinks it is based on its odometry
        p = PoseStamped(header=Header(stamp=data.header.stamp, frame_id=self.base_frame), pose=Pose())
        self.odom_pose = self.tf_listener.transformPose(self.odom_frame, p)

        # we need to be able to compare the current odom pose to the prior odom pose
        # if there isn't a prior odom pose, set the odom_pose variable to the current pose
        if not self.odom_pose_last_motion_update:
            self.odom_pose_last_motion_update = self.odom_pose
            return

        # check to see if we've moved far enough to perform an update
        curr_x = self.odom_pose.pose.position.x
        old_x = self.odom_pose_last_motion_update.pose.position.x
        curr_y = self.odom_pose.pose.position.y
        old_y = self.odom_pose_last_motion_update.pose.position.y
        curr_yaw = get_yaw_from_pose(self.odom_pose.pose)
        old_yaw = get_yaw_from_pose(self.odom_pose_last_motion_update.pose)
        yaw_diff = np.abs(curr_yaw - old_yaw)
        if yaw_diff > np.pi:
            yaw_diff = np.pi * 2 - yaw_diff

        if (np.abs(curr_x - old_x) > self.lin_mvmt_threshold or
            np.abs(curr_y - old_y) > self.lin_mvmt_threshold or
                yaw_diff > self.ang_mvmt_threshold):

            # This is where the main logic of the particle filter is carried out
            self.update_particles_with_motion_model()
            self.update_particle_weights_with_measurement_model(data)
            self.normalize_particles()
            self.resample_particles()
            self.update_estimated_robot_pose()
            self.publish_particle_cloud()
            self.publish_estimated_robot_pose()

            self.odom_pose_last_motion_update = self.odom_pose

    def update_estimated_robot_pose(self):
        # based on the particles within the particle cloud, update the robot pose estimate
        x_sum = 0
        y_sum = 0
        ang_sum = 0
        for p in self.particle_cloud:
            x_sum += p.pose.position.x
            y_sum += p.pose.position.y
            ang_sum += get_yaw_from_pose(p.pose)
        self.robot_estimate = Pose(Point(x_sum / self.num_particles, y_sum / self.num_particles, 0), Quaternion(*quaternion_from_euler(0, 0, ang_sum / self.num_particles)))

    def update_particle_weights_with_measurement_model(self, data):
        for particle in self.particle_cloud:
            q = 1
            ang = get_yaw_from_pose(particle.pose)
            for i in range(0, 360, self.delta_ang):
                z = data.ranges[i]
                if not data.range_min <= z <= data.range_max:
                    continue
                x = particle.pose.position.x + z * np.cos(i / 180 * np.pi + ang)
                y = particle.pose.position.y + z * np.sin(i / 180 * np.pi + ang)
                dist = self.likelihood_field.get_closest_obstacle_distance(x, y)
                if np.isnan(dist):
                    dist = self.default_dist
                q *= self.z_hit * compute_prob_zero_centered_gaussian(dist, self.sigma_dist) + self.z_random / data.range_max
            particle.w = q

    def update_particles_with_motion_model(self):
        # based on the how the robot has moved (calculated from its odometry), we'll  move
        # all of the particles correspondingly
        p1 = self.odom_pose_last_motion_update.pose
        x1 = p1.position.x
        y1 = p1.position.y
        theta1 = get_yaw_from_pose(p1)
        p2 = self.odom_pose.pose
        x2 = p2.position.x
        y2 = p2.position.y
        theta2 = get_yaw_from_pose(p2)
        r1 = np.arctan2(y2 - y1, x2 - x1) - theta1
        t = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        r2 = theta2 - theta1 - r1
        for particle in self.particle_cloud:
            r1_ = r1 + np.random.standard_normal() * self.sigma_r
            t_ = t + np.random.standard_normal() * self.sigma_t
            r2_ = r2 + np.random.standard_normal() * self.sigma_r
            p = particle.pose
            theta = get_yaw_from_pose(p)
            p.position.x += t_ * np.cos(theta + r1_)
            p.position.y += t_ * np.sin(theta + r1_)
            p.orientation = Quaternion(*quaternion_from_euler(0, 0, theta + r1_ + r2_))


if __name__ == "__main__":
    ParticleFilter()
    rospy.spin()
