import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation 
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    omega = np.zeros(3)
    #
    omega_1 = np.zeros(3)
    # Compute the rotation matrix
    # R_curr is the Transformation matrix from current orientation to the world frame
    # R_des is the Transformation matrix from  desired orientation to the world frame
    # Hence, we have to invert R_curr and multiply it with R_des, to give the transformation matrix from desired to current
    R_diff = np.dot(R_curr.T, R_des)
    
    # Calculate the angle using the trace of R_diff
    angle = np.arccos((np.trace(R_diff) - 1) / 2)
    
    if angle == 0:
        # No rotation needed
        return np.zeros(3)

    # Compute the skew-symmetric part
    S = (R_diff - R_diff.T) / 2
    
    # Extract the axis of rotation (the coefficients of the skew-symmetric matrix)
    omega_1 = np.array([S[2, 1], S[0, 2], S[1, 0]])

    # We have to multiply omega_1 with the R_curr matrix to give the omega in the world frame
    omega = np.dot(R_curr,omega_1)

    return omega 

