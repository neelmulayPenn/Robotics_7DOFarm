import numpy as np
from math import pi

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout
        self.d = [0.141, 0.192, 0, 0.316, 0, 0.384, 0, 0.21]
        self.a = [0, 0, 0, -0.0825, 0.0825, 0, 0.088, 0]
        self.alpha = [0, pi/2, -pi/2, -pi/2, pi/2, -pi/2, pi/2, 0]

        # Calibrate some A matrix for joint position calculation
        # Due to the DH convension, some intermediate frame origins may not lie on
        # the conrresponding joint centers. These matrices below are to calibrate the 
        # frame origins to the joint centers to get the correct pose (Position and orientation) 
        # of the joint expressed in the base frame.
        self.calibration_matrices = {
            2 : np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.195],
                [0, 0, 0, 1]
            ]),

            4 : np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.125],
                [0, 0, 0, 1]
            ]),

            5 : np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -0.015],
                [0, 0, 0, 1]
            ]),

            6 : np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.051],
                [0, 0, 0, 1]
            ])      
        }

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here

        # The DH parameter: thetas, as variables of each joint angle.
        theta = [0, q[0]+pi, q[1], q[2], q[3], q[4], q[5]-pi, q[6]-pi/4]
        A = self.compute_Ai(theta)
        # We can get (n+1) As here. An extra A_0 is computed to account for the first joint position w.r.t the base frame. 

        # The end effector homogeneous transformation matrix
        T0e = A[0]
        for i in range(1, len(A)):
            T0e = np.matmul(T0e, A[i])

        T = self.compute_T0i(A)  
        # print(T.shape) # output: (8, 4, 4)

        # Tev is the transformation matrix from the end effector to the virtual joint 1, with R = I and T = [0, 0.1, -0.105]
        Tev = np.array([    [1, 0, 0, 0],   
                            [0, 1, 0, 0.1],
                            [0, 0, 1, -0.105],
                            [0, 0, 0, 1]    ])
        
        # Tew is the transformation matrix from the end effector to the virtual joint 2, with R = I and T = [0, -0.1, -0.105]
        Tew = np.array([    [1, 0, 0, 0],
                            [0, 1, 0, -0.1],
                            [0, 0, 1, -0.105],
                            [0, 0, 0, 1]    ])
        
        T0v = np.matmul(T0e, Tev)
        T0w = np.matmul(T0e, Tew)

        # Add T0v and T0w to T
        T = np.concatenate((T, [T0v], [T0w]), axis=0)

        # print(T.shape) # Expected to be (10, 4, 4), but get 160
        
        
        T0e = T # T0e is now a 10x4x4 matrix, with each 4x4 matrix representing the transformation matrix of each joint/end effector frame expressed in the world frame
        # The order is: 7 real joins, end effector, virtual joint 1, virtual 2
        jointPositions = np.array([T[i][:3, 3] for i in range(len(T))])
        # print("The jointPositions.shape returned by CalculateFKJac is ",jointPositions.shape) # Expected to be (10, 3), but get (160, 3

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1


    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame (deprecated)

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        # theta = [0, q[0]+pi, q[1], q[2], q[3], q[4], q[5]-pi, q[6]-pi/4]
        # A = self.compute_Ai(theta)
        # T = self.compute_T0i(A)
        _, T = self.forward_expanded(q)

        rotation_axis = []
        
        for i in range(len(T)):
            rotation_axis.append(T[i][:3, 2])

        axis_of_rotation_list = np.column_stack(rotation_axis)

        return axis_of_rotation_list
    
    
    def compute_T0i(self, A):
        T = []
        # Compute T[i] = A[0] @ A[1] @ ... @ A[i] where T[i] is T0i
        for i in range(len(A)):
            # Start by setting the first T[0] as A[0]
            if i == 0:
                T.append(A[0])
            else:
                # Multiply A[0] @ A[1] @ ... @ A[i]
                T.append(T[i - 1] @ A[i])


        for calibation_index in self.calibration_matrices:
            T[calibation_index] = T[calibation_index] @ self.calibration_matrices[calibation_index] 
        T = np.array(T)
        # print("This is the shape of T in compute_T0i function",T.shape)
        return(T)




    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE
        A = []
        for i in range(len(q)):
            A_i = np.array([
                [np.cos(q[i]), -np.sin(q[i]) * np.cos(self.alpha[i]),  np.sin(q[i]) * np.sin(self.alpha[i]), self.a[i] * np.cos(q[i])],
                [np.sin(q[i]),  np.cos(q[i]) * np.cos(self.alpha[i]), -np.cos(q[i]) * np.sin(self.alpha[i]), self.a[i] * np.sin(q[i])],
                [0,             np.sin(self.alpha[i]),                np.cos(self.alpha[i]),                self.d[i]],
                [0,             0,                                     0,                                     1]
            ])
            A.append(A_i)

        # for i in range(len(A)):
        #     print(f"A{i}",A[i])

        return(A)


    def compute_vel_FK(self, q):
        # joint_positions, T0e = self.forward_expanded(q)
        # axis_of_rotation_list = self.get_axis_of_rotation(q) # (3, 7) matrix
        # o_e = joint_positions[-1] # (3, ) vector
        # joint_origins = [] # (7, 3) matrix
        # for i in range(len(joint_positions)-1):
        #     joint_origins.append(joint_positions[i])

        # J_v = []
        # J_w = []

        # for i in range(len(joint_origins)):
        #     z_i = axis_of_rotation_list[:, i]
        #     o_i = joint_origins[i]
        #     J_vi = np.cross(z_i, (o_e - o_i))
        #     J_wi = z_i
        #     J_v.append(J_vi)
        #     J_w.append(J_wi)
        # J_v = np.column_stack(J_v)
        # J_w = np.column_stack(J_w)
        # J = np.row_stack((J_v, J_w))

        # Calculate Jvi for each joint (real and virtual) and end effector
        # No need to calculate Jv0 since it is always 0 for the first joint
        # Don't care about Jw
        # Calculate the joint position and the transformation matrix for each joint at current configuration
        joint_positions, T = self.forward_expanded(q)
        # Calculate the axis of rotation for each joint
        axis_of_rotation_list = self.get_axis_of_rotation(q)
        J_v = []

        # Calculate Jvi for each joint, Jvi will always be of size 3x9, with i non-zero columns (corresponding to joints before it) and 9-1 zero columns (corresponding to itself and joints (end effector) after it)
        for i in range(len(joint_positions)):
            Jvi = []
            # Calculate every column Jvi_j for each joint j before i
            o_i = joint_positions[i]
            for j in range(i):
                z_j = axis_of_rotation_list[:, j]
                o_j = joint_positions[j]
                Jvi_j = np.cross(z_j, (o_i - o_j))
                Jvi.append(Jvi_j)
            # Add zero columns for the joint i, joints after it and the end effector
            Jvi += [np.zeros(3)] * (len(joint_positions) - i)
            # Make Jvi is 3x9, note that Jv0 is a 3x9 matrix with all zeros
            Jvi = np.column_stack(Jvi)
            J_v.append(Jvi) # J_v is of length 10, each element is a 3x9 Jvi matrix.
        return J_v
    
if __name__ == "__main__":

    fk = FK_Jac()

    # # matches figure in the handout
    # q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    # joint_positions, T0e = fk.forward_expanded(q)
    
    # print("Joint Positions:\n",joint_positions)
    # print("End Effector Pose:\n",T0e)

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    # q = np.array([0,0,0,0,0,0,0])

    joint_positions, T0e = fk.forward_expanded(q)

    jacobian_matrix = fk.compute_vel_FK(q)

    q_dot = [0, 0, 1, 0, 0, 0, 0]

    twist = np.matmul(jacobian_matrix, q_dot)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
    print("Jacobian matrix:\n",jacobian_matrix)
    print("Twist of the end effector:\n",twist)
