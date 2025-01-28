import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))
    
    dq1= np.zeros((1, 7))
    dq2= np.zeros((1, 7))

    # We calculate the Jacobian at the current configuration
    J = calcJacobian(q_in)
    
    # Combine linear and angular velocity into one 6x1 target velocity vector
    # Stacking the linear velocity and the angular velocity 
    vel_target = np.vstack((v_in, omega_in))
    
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
    #dq = dq1.squeeze()
    
    I = np.identity(7)
    #J_pseudoinv, _, _, _ = np.linalg.lstsq(J_reduced, J, rcond=None)
    #null = np.dot((I - J_pseudoinv),b)
    
    # Calculate null space projection matrix
    #J_pseudoinv, _, _, _ = np.linalg.lstsq(J_reduced, J, rcond=None)
    #null_space_proj = I - np.dot(J_pseudoinv, J)

    # Apply the null-space projection to b
    #null = np.dot(null_space_proj, b).squeeze()
    #null = np.dot((I - J_pseudoinv),b).squeeze()
    
    # Null-space projection for secondary task
    #I = np.identity(7)
    J_pseudoinv = np.linalg.pinv(J_reduced)  # Pseudo-inverse of J for projection
    dq=np.dot(J_pseudoinv,vel_reduced).squeeze()
    null_space_proj = I - np.dot(J_pseudoinv, J_reduced)
    null = np.dot(null_space_proj, b).squeeze()

    return dq + null

