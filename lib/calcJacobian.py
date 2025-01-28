import numpy as np
from lib.calculateFK import FK 

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    # Initialize Jacobian matrix - It is 6x7 because the FRANKA robot is 7 DOF
    J = np.zeros((6, 7))
    T0e = np.identity(4)
    fk = FK()  # Create an instance of the FK class
    jointPositions, FinalTransfMat = fk.forward(q_in)  # Compute forward kinematics
    
    # The origin of the end-effector in the world frame
    o_n = FinalTransfMat[:3, 3]

    # Initialize the z-axis of the base frame (for i=0)
    z_prev = np.array([0, 0, 1])  # Base frame z-axis
    p_prev = np.array([0, 0, 0.141])  # Base frame origin (joint 1 base)

    # Loop through each joint
    for i in range(len(fk.dh_params)):
        a, alpha, d, theta_offset = fk.dh_params[i]
        A = fk.dh_transform(a, alpha, d, q_in[i] + theta_offset)
        T0e = T0e @ A  # Update T0e to current joint's frame

        # Compute the current joint position
        joint_pos = T0e @ fk.offset_joint_positions[i+1]
        p_i = joint_pos[:3]  # Extract the current joint position (x, y, z)

        # Extract the z-axis of the current joint
        z_i = T0e[:3, 2]

        # Use z_prev to calculate linear velocity
        J_v = np.cross(z_prev, o_n - p_prev)

        # The angular velocity component
        J_w = z_prev

        # Update the Jacobian
        J[:3, i] = J_v  # Linear velocity part
        J[3:, i] = J_w  # Angular velocity part

        # Update p_prev and z_prev for the next iteration
        p_prev = p_i
        z_prev = z_i

    ## STUDENT CODE GOES HERE

    return J

if __name__ == '__main__':
    q = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
