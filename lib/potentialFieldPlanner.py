import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from calculateFKJac import FK_Jac
from detectCollision import detectCollision
from loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()
    attractive_strength = [30,30,30,30,30,30,30,100,100,100]
    repulsive_strength = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    attractive_threshold = 0.012  # Threshold distance for switching between conic and parabolic potential
    repulsive_range = 0.012

    def __init__(self, tol=1e-2, max_steps=2000, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

        


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current,i):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(current - target)
        if distance > PotentialFieldPlanner.attractive_threshold:
            # Conic potential field
            att_f = -(current - target) / distance
        else:
            # Parabolic potential field
            att_f = -PotentialFieldPlanner.attractive_strength[i] * (current - target)

        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, i, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE

        # Calculate distance and use the provided unit vector if given
        dist, unit = PotentialFieldPlanner.dist_point2box(current.reshape(1, 3), obstacle)
        distance = dist[0]
        unit_vector = unitvec if np.linalg.norm(unitvec) != 0 else unit[0].reshape((3, 1))

        # If outside the range of influence, the repulsive force is zero
        # if distance > PotentialFieldPlanner.repulsive_range:
        #     rep_f = np.zeros((3, 1))
        # else:
        #     # Compute the repulsive force using the provided formula
        #     rep_f = PotentialFieldPlanner.repulsive_strength[i] * (1.0 / distance - 1.0 / PotentialFieldPlanner.repulsive_range) * (1.0 / (distance ** 2)) * (-unit_vector)

        if distance > 0 and distance <= PotentialFieldPlanner.repulsive_range:
            rep_f = PotentialFieldPlanner.repulsive_strength[i] * (1.0 / distance - 1.0 / PotentialFieldPlanner.repulsive_range) * (1.0 / (distance ** 2)) * (-unit_vector)
        else:
            rep_f = np.zeros((3, 1))
        ## END STUDENT CODE
        rep_f = rep_f.flatten()

        # print("rep_f shape: ", rep_f.shape)

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE

        # joint_forces = np.zeros((3, 9)) 
        joint_forces = np.zeros((3, 9))

        for i in range(9):
            # Compute the attractive force
            attractive_force = PotentialFieldPlanner.attractive_force(target[:, i], current[:, i], i+1)
            
            # Compute the repulsive forces
            repulsive_force = np.zeros(3)
            for j in range(len(obstacle)):
                # print(current[:, i].shape)
                repulsive_force += PotentialFieldPlanner.repulsive_force(obstacle[j], current[:, i], i+1)
            
            # Sum the forces
            # print("The shape of the attractive_force for one joint is",attractive_force.shape)
            # print("The shape of the repulsive_force for one joint is",repulsive_force.shape)
            joint_forces[:, i] = attractive_force + repulsive_force
                
        ## END STUDENT CODE

        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE

        joint_torques = np.zeros((1, 9)) 
        # Calculate the Jacobians of the joints in current config
        Jv = PotentialFieldPlanner.fk.compute_vel_FK(q) # Jv is of shape 10x3x9, where 10 is the number of "end effectors" (7 real joints + 2 virtual joints + 1 end effector)
        # Jv_0 = 0 because the very first joint is fixed
        # So we only care about the last nine joints/end effector, each feels a force in the potential field in shape of 3x1. 
        # Each force can be converted to a torque vector by pre multiplying the transpose of the Jacobian of the joint/end effector. (9x3)x(3x1) = 9x1
        # Summing up all the torques on each joint/end effector gives the total torque on each joint/end effector.
        for i in range(9): 
            Jv_i = Jv[i+1] 
            Jv_i = Jv_i[:, 1:] # Jv_i is of shape 3x9
            # print("The shape of Jv_i is",Jv_i.shape)
            joint_forces_i = joint_forces[:, i] # joint_forces_i is of shape 3x1
            # print("The shape of joint_forces_i is",joint_forces_i.shape)
            joint_torques += np.dot(Jv_i.T, joint_forces_i).T
        
        
        # joint_torques = joint_torques[:, :7]


        ## END STUDENT CODE

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        # distance = 0
        distance = np.linalg.norm(target - current)

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """

        ## STUDENT CODE STARTS HERE

        dq = np.zeros((1, 7))
        obstacle = map_struct.obstacles
        # q = q.flatten()
        joint_position_curr, T_curr = PotentialFieldPlanner.fk.forward_expanded(q)
        joint_position_curr = joint_position_curr[1:, :].T
        joint_position_target, T_target = PotentialFieldPlanner.fk.forward_expanded(target)
        joint_position_target = joint_position_target[1:, :].T
        # Compute the forces on each joint
        joint_forces = PotentialFieldPlanner.compute_forces(joint_position_target, obstacle, joint_position_curr)
        # Compute the torques on each joint
        joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)
        # Compute the gradient
        joint_torques = joint_torques[:, :7]
        # dq = joint_torques/np.linalg.norm(joint_torques)
        # alternative:
        joint_torques_magic = joint_torques - 5 * (q-target)
        dq = joint_torques_magic/np.linalg.norm(joint_torques_magic)
        ## END STUDENT CODE

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        q_path = np.array([]).reshape(0,7)
        obstacle = map_struct.obstacles
        q = start.copy().reshape(1, 7)
        q_path = np.vstack((q_path, start))
        iterations = 0

        # Check if start or goal configuration is in collision
        joint_positions_start, _ = PotentialFieldPlanner.fk.forward_expanded(start)
        joint_positions_goal, _ = PotentialFieldPlanner.fk.forward_expanded(goal)
        for obs in obstacle:
            if np.array(detectCollision(joint_positions_start[:-1], joint_positions_start[1:], obs)).any() or np.array(detectCollision(joint_positions_goal[:-1], joint_positions_goal[1:], obs)).any():
                return q_path  # Return empty path if start or goal is in collision

        while iterations < self.max_steps:
            
            # Compute gradient
            dq = self.compute_gradient(q.flatten(), goal, map_struct)



            # Update joint configuration
            step_size = 0.03  # You may need to tune this step size
            q_new = q + step_size * dq

            # Clip the new joint configuration to respect joint limits
            q_new = np.clip(q_new, PotentialFieldPlanner.lower, PotentialFieldPlanner.upper)

            # Interpolate points between current q and new q to check for collisions
            num_intermediate = 20  # Number of intermediate points to check for collision
            collision_detected = False
            for alpha in np.linspace(0, 1, num_intermediate):
                q_interp = (1 - alpha) * q + alpha * q_new
                joint_positions_interp, _ = PotentialFieldPlanner.fk.forward_expanded(q_interp.flatten())
                for obs in obstacle:
                    if np.array(detectCollision(joint_positions_interp[:-1], joint_positions_interp[1:], obs)).any():
                        collision_detected = True
                        break
                if collision_detected:
                    break
            random_walk_tries = 0
            while random_walk_tries < 10000 and (collision_detected or np.linalg.norm(dq) < self.min_step_size):
                random_walk_tries += 1
                random_magnitude = np.random.uniform(low=-0.1, high=5.0)
                options = np.array([-random_magnitude, random_magnitude])
                random_dq = np.random.choice(options, size=(7,))
                q_ran = q_new + random_dq
                q_ran = np.clip(q_ran, PotentialFieldPlanner.lower, PotentialFieldPlanner.upper)
                dq = self.compute_gradient(q_ran.flatten(), goal, map_struct)
                q_new = q_ran + step_size * dq
                q_new = np.clip(q_new, PotentialFieldPlanner.lower, PotentialFieldPlanner.upper)
                collision_detected = False
                for alpha in np.linspace(0, 1, num_intermediate):
                    q_interp = (1 - alpha) * q_ran + alpha * q_new
                    joint_positions_interp, _ = PotentialFieldPlanner.fk.forward_expanded(q_interp.flatten())
                    for obs in obstacle:
                        if np.array(detectCollision(joint_positions_interp[:-1], joint_positions_interp[1:], obs)).any():
                            collision_detected = True
                            break
                    if collision_detected:
                        break
                

            # if collision_detected or np.linalg.norm(dq) < self.min_step_size:
            #     # Generate random walk step if collision detected
            #     random_walk_attempts = 0
            #     max_random_walks = 50
            #     while random_walk_attempts < max_random_walks:
            #         dq = np.random.randn(1, 7)
            #         q_new = q + step_size * dq
            #         q_new = np.clip(q_new, PotentialFieldPlanner.lower, PotentialFieldPlanner.upper)
            #         collision_free = True
            #         for alpha in np.linspace(0, 1, num_intermediate):
            #             q_interp = (1 - alpha) * q + alpha * q_new
            #             joint_positions_interp, _ = PotentialFieldPlanner.fk.forward_expanded(q_interp.flatten())
            #             for obs in obstacle:
            #                 if np.array(detectCollision(joint_positions_interp[:-1], joint_positions_interp[1:], obs)).any():
            #                     collision_free = False
            #                     break
            #             if not collision_free:
            #                 break
            #         if collision_free:
            #             break
            #         random_walk_attempts += 1
            #     else:
            #         continue
            
            # Append new configuration to path if no collision is found
            q_path = np.vstack((q_path, q_new))
            q = q_new

            # Termination condition: if distance to goal is less than tolerance
            if self.q_distance(q.flatten(), goal) < self.tol:
                break

            print(f"Iteration: {iterations}, Distance to goal: {self.q_distance(q.flatten(), goal)}")

            # Increase iteration count
            iterations += 1



        # while True:

        #     ## STUDENT CODE STARTS HERE
            
        #     # The following comments are hints to help you to implement the planner
        #     # You don't necessarily have to follow these steps to complete your code 
            
        #     # Compute gradient 
        #     # TODO: this is how to change your joint angles 

        #     # Termination Conditions
        #     if True: # TODO: check termination conditions
        #         break # exit the while loop if conditions are met!

        #     # YOU NEED TO CHECK FOR COLLISIONS WITH OSTACLES
        #     # TODO: Figure out how to use the provided function 

        #     # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
        #     # TODO: when detect a local minima, implement a random walk
            
        #     ## END STUDENT CODE

        return q_path

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("/Users/neelmulay/Documents/Penn/MEAM 5200/Labs/lib/maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
