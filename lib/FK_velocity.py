import numpy as np 
from lib.calcJacobian import calcJacobian   # type: ignore

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """

    ## STUDENT CODE GOES HERE

    velocity = np.zeros((6, 1))

    # Compute the Jacobian
    J = calcJacobian(q_in)

    # Multiply the Jacobian by joint velocities
    velocity = J @ dq

    return velocity
