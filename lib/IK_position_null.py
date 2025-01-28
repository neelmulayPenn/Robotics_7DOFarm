import numpy as np
from math import pi, acos
from scipy.linalg import null_space

from lib.calcJacobian import calcJacobian 
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff
from lib.IK_velocity import IK_velocity  #optional
from lib.IK_velocity_null import IK_velocity_null  #optional


class IK:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=1000, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size
        


    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """

        ## STUDENT CODE STARTS HERE
        axis1 = np.zeros(3)

        #Calculating the displacement from current to target, as per the formula in the slides
        displacement = current[:3,3] - target[:3,3]
        
        # Compute the rotation matrix
        # Hence, we have to invert R_curr and multiply it with R_des, to give the transformation matrix from desired to current
        R_curr = np.zeros(3)
        R_des = np.zeros(3)
        # R_curr is the Transformation matrix from current orientation to the world frame
        R_curr = current[0:3, 0:3]
        # R_des is the Transformation matrix from  desired orientation to the world frame
        R_des = target[0:3, 0:3]
        # Using the same analogy, we find the rotation from current to target
        R_diff = np.dot(R_des.T, R_curr)
        
        # Calculate the angle using the trace of R_diff
        # Since we are using acos, we need to clip the values to stay in the correct range
        angle = acos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))

        if angle == 0:
            axis = np.zeros(3)
            return displacement, axis

        # Compute the skew-symmetric part
        S = (R_diff - R_diff.T) / 2
        
        # Extract the axis of rotation (the coefficients of the skew-symmetric matrix)
        axis1 = np.array([S[2, 1], S[0, 2], S[1, 0]])
        
        # We have to multiply axis1 with the R_curr matrix to give the omega in the world frame
        axis = np.dot(R_curr,axis1)
        
        ## END STUDENT CODE
        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H
        """
        
        ## STUDENT CODE STARTS HERE
        
        # Finding the displacement from G to H
        displacement_vec = G[:3, 3] - H[:3, 3]
        distance=np.linalg.norm(displacement_vec)
        
        # To find the angle, we compute the rotation  matrix
        rotation_matrix = np.dot(G.T,H)
        # We get the angle using the trace formula of a rotation matrix
        trace_value = np.clip((np.trace(rotation_matrix) - 1) / 2, -1, 1)
        angle = acos(trace_value)
        
        ## END STUDENT CODE
        return distance, angle

    def is_valid_solution(self,q,target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """
        
        ## STUDENT CODE STARTS HERE
        # Initializing the success to be true at the beginning
        success = True
        message = "Solution found"

        for i in range(len(q)):
            # If any of the joints are outside the limit
            if(q[i]>=self.upper[i] or q[i]<=self.lower[i]):
                success = False
                message = f"Solution found/not found + Joint angle {i} out of bounds."
                return success,message
        
        # We find the achieved transformation matrix using the joint angles.
        _,achieved = self.fk.forward(q)

        # We find the distance and angle between the achieved and the target orientations
        distance, angle = self.distance_and_angle(achieved, target)
        
        # Check if the pose is within linear tolerance
        if distance > self.linear_tol:
            success = False
            message = f"Solution found/not found + distance {distance} > distance tolerance {self.linear_tol}."
            return success, message

        # Check if the pose is within angular tolerance
        if angle > self.angular_tol:
            success = False
            message = f"Solution found/not found + Angle {angle} > angle tolerance {self.angular_tol}."
            return success, message

        return success, message

        
        ## END STUDENT CODE


    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q,target,method):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (J pseudo-inverse or J transpose) in your algorithm
        
        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        ## STUDENT CODE STARTS HERE
        dq = np.zeros(7)
        
        _,currentPose = IK.fk.forward(q)
        # error = currentOrientation - targetOrientation
        
        displacement,axis = IK.displacement_and_axis(target,currentPose)
        v_lin = displacement.reshape((3, 1))  # Linear velocity to close the positional gap
        v_ang = axis.reshape((3, 1))     # Angular velocity to close the orientation gap
        # Stacking them to get a complete velocity vector
        vel_target = np.vstack((v_lin, v_ang)).flatten() 
        
        J=calcJacobian(q)
        J_pseudo = np.linalg.pinv(J)  
        #J_v_pseudo = J_pseudoinv[0:3, :]
        
        # We use the Pseudoinverse of J or J transpose depending on the value stored by method
        # Negative dq, so that it is in line with the secondary task, which is also negative
        if method=='J_pseudo':
            dq = -np.dot(J_pseudo,vel_target)
        elif method=='J_trans':
            dq = -np.dot(J.T,vel_target)

        ## END STUDENT CODE
        return dq

    @staticmethod
    def joint_centering_task(q,rate=5e-1): 
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset # proportional term (implied quadratic cost)

        return dq
        
    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed, method, alpha):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (J pseudo-inverse or J transpose) in your algorithm

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = seed
        rollout = []

        ## STUDENT CODE STARTS HERE
        v_in = np.zeros((3,1))
        omega_in = np.zeros((3,1))
        steps=0
        I = np.identity(7)
        
        ## gradient descent:
        while True:
            rollout.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = IK.end_effector_task(q,target, method)
            
            J = calcJacobian(q)
            J_pseudoinv = np.linalg.pinv(J)   
            
            # Secondary Task - Center Joints
            dq_center = IK.joint_centering_task(q)
            
            ## Task Prioritization
            # Using the null space of the Jacobian J
            # The null space of J is used, so that the second task does not interfere with the first task
            dot1 = np.dot(J_pseudoinv,J)
            nullspace_proj = I - dot1
            # We compute the final dq 
            dq = dq_ik + np.dot(nullspace_proj,dq_center)
            
            # Check termination conditions
            # Check convergence - termination conditions
            if steps >= self.max_steps:
                break
            elif np.linalg.norm(dq) < self.min_step_size:
                break
            #break

            # update q
            # We are adding, although it is gradient DESCENT, because dq is negative
            q += alpha * dq
            steps=steps+1

        ## END STUDENT CODE

        success, message = self.is_valid_solution(q,target)
        return q, rollout, success, message

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    ik = IK()

    # matches figure in the handout
    seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    target = np.array([
        [0,-1,0,-0.2],
        [-1,0,0,0],
        [0,0,-1,.5],
        [0,0,0, 1],
    ])

    # Using pseudo-inverse 
    q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target, seed, method='J_pseudo', alpha=.5)

    for i, q_pseudo in enumerate(rollout_pseudo):
        joints, pose = ik.fk.forward(q_pseudo)
        d, ang = IK.distance_and_angle(target,pose)
        print('iteration:',i,' q =',q_pseudo, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # Using pseudo-inverse 
    q_trans, rollout_trans, success_trans, message_trans = ik.inverse(target, seed, method='J_trans', alpha=.5)

    for i, q_trans in enumerate(rollout_trans):
        joints, pose = ik.fk.forward(q_trans)
        d, ang = IK.distance_and_angle(target,pose)
        print('iteration:',i,' q =',q_trans, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # compare
    print("\n method: J_pseudo-inverse")
    print("   Success: ",success_pseudo, ":  ", message_pseudo)
    print("   Solution: ",q_pseudo)
    print("   #Iterations : ", len(rollout_pseudo))
    print("\n method: J_transpose")
    print("   Success: ",success_trans, ":  ", message_trans)
    print("   Solution: ",q_trans)
    print("   #Iterations :", len(rollout_trans),'\n')
