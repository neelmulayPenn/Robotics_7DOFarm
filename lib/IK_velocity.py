import numpy as np 
from lib.calcJacobian import calcJacobian

def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq 
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros((1, 7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))

    dq1= np.zeros((1, 7))

    # We calculate the Jacobian at the current configuration
    J = calcJacobian(q_in)
    
    # Combine linear and angular velocity into one 6x1 target velocity vector
    # Stacking the linear velocity and the angular velocity 
    vel_target = np.vstack((v_in.reshape((3,1)), omega_in.reshape((3,1))))
    
    # We must handle NaN values in vel_target by removing corresponding rows from Jacobian and target velocity
    # Mask for non-NaN values
    mask = ~np.isnan(vel_target)  
    # Remove rows where target velocity is unconstrained
    J_reduced = J[mask.flatten(), :]  
    # Remove corresponding entries from target velocity
    vel_reduced = vel_target[mask]  

    # Using least squares method to solve for joint velocities
    # np.linalg.lstsq returns a tuple so we must handle for the same, by taking only the first value 
    dq1, _, _, _ = np.linalg.lstsq(J_reduced, vel_reduced, rcond=None)
    
    # Converting to a 1D form, returning the 1x7 joint velocity vector 
    dq = dq1.squeeze()

    return dq
