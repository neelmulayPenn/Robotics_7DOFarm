import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        # Define the DH parameters as class attributes: [a, alpha, d, theta_offset]
        self.dh_params = [
            [0, -pi/2, 0.333, 0],          # Joint 1
            [0, pi/2, 0, 0],               # Joint 2
            [0.0825, pi/2, 0.316, 0],      # Joint 3
            [0.0825, pi/2, 0, pi],         # Joint 4
            [0, -pi/2, 0.384, 0],          # Joint 5
            [0.088, pi/2, 0, -pi],         # Joint 6
            [0, 0, 0.210, -pi/4]           # Joint 7
        ]
        
        # Offset positions for each joint (in homogeneous coordinates)
        self.offset_joint_positions = np.array([
            [0, 0, 0.141, 1],  # Joint 1
            [0, 0, 0, 1],      # Joint 2
            [0, 0, 0.195, 1],  # Joint 3
            [0, 0, 0, 1],      # Joint 4
            [0, 0, 0.125, 1],  # Joint 5
            [0, 0, -0.015, 1], # Joint 6
            [0, 0, 0.051, 1],  # Joint 7
            [0, 0, 0, 1]       # End effector (EE)
        ])

        pass

    def dh_transform(self, a, alpha, d, theta):
        # Compute the transformation matrix Ai which transforms from frame i-1 to i
        A = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0,             np.sin(alpha),                 np.cos(alpha),                d],
            [0,             0,                             0,                            1]
        ])
        return A 

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)

        jointPositions[0] = [0, 0, 0.141]  # Base position for joint 1 in the world frame

        # Loop over each joint and apply the transformations
        for i in range(len(self.dh_params)):
            a, alpha, d, theta_offset = self.dh_params[i]
            A = self.dh_transform(a, alpha, d, q[i] + theta_offset)  # Compute transformation
            T0e = T0e @ A  # Update T0e with the current transformation matrix
            joint_pos = T0e @ self.offset_joint_positions[i+1].reshape(4, 1)  # Compute joint position
            jointPositions[i+1] = joint_pos[:3, 0]  # Extract (x, y, z) position

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    #q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    q = np.array([pi/2,0,0,0,0,0,0])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)


    timeSum = np.array(10)
    iterations = np.array(10)
     