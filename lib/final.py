import sys
import numpy as np
from copy import deepcopy
from math import pi
import threading
from queue import Queue
import time

import rospy
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from core.utils import time_in_seconds
from lib.IK_position_null import IK
from lib.calculateFK import FK

class IKThread(threading.Thread):
    def __init__(self, pose_0, seed, pose_queue):
        threading.Thread.__init__(self)
        self.pose_0 = pose_0
        self.seed = seed
        self.pose_queue = pose_queue
        self.ik = IK()
    
    def run(self):
        # Compute IK solutions for both positions (over target and at target)
        target_adj, yaw = fix_pose(self.pose_0)
        q_adjust = np.array([0, 0, 0, 0, 0, 0, yaw])
        
        # Compute over target position
        q_over_target, _, _, _ = self.ik.inverse(target_adj, self.seed, method='J_pseudo', alpha=.5)
        q_over_target = q_over_target - q_adjust
        
        # Check joint limits
        if (q_over_target[6] > 2.8973):
            q_over_target[6] = q_over_target[6] - pi
        if (q_over_target[6] < -2.8973):
            q_over_target[6] = q_over_target[6] + pi
            
        # Put results in queue
        self.pose_queue.put((self.pose_0, q_over_target, target_adj, q_adjust))

def fix_pose(actual):
    z_shift = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0.1],
                       [0, 0, 0, 0]])
    bias = np.array([[0, -1, 0],
                     [-1, 0, 0],
                     [0, 0, -1],
                     [0, 0, 0]])
    # extract position of cube and put into biased orientation
    yaw = np.arctan2(actual[1,0], actual[0,0])
    pos = actual[:,3].reshape((4,1))
    print(bias)
    print(pos)
    bias = np.hstack((bias, pos))
    bias = bias + z_shift # shift above cube
    return bias, yaw

def move_over_target(target, current):
        """
        INPUTS:
            target: 4x4 homogeneous tranformation matrix of target (block)
            current: 1x7 current joint configuration
        
        OUTPUTS:
            q_over_target: 1x7 target (now current) joint configuration
            target_adj: 4x4 adjusted target (now current) homogeneous matrix of pose where the end-effector is 10 cm over the target block

        This method moves the arm from the current pose to a pose where the end-effector is 10 cm over the target center
        """
        target_adj, yaw = fix_pose(target) # adjust desired orientation and find position 5 cm above target
        q_adjust = np.array([0, 0, 0, 0, 0, 0, yaw])
        q_over_target, _, _, _ = ik.inverse(target_adj, current, method='J_pseudo', alpha=.5)  #try both methods (pseudo is way faster)
        q_over_target = q_over_target - q_adjust # adjust end effector orientation to align with block
        # check joint limits:
        if (q_over_target[6] > 2.8973):
            q_over_target[6] = q_over_target[6] - pi
        if (q_over_target[6] < -2.8973):
            q_over_target[6] = q_over_target[6] + pi
        arm.safe_move_to_position(q_over_target) # move gripper 5 cm over target cube with biased end effector rotation
        arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH, 0) # open gripper 10 cm with zero force
        return q_over_target, target_adj, q_adjust
    
def pick_up_target(target, current, q_adjust, q_return):
        q_target, _, _, _ = ik.inverse(target, current, method='J_pseudo', alpha=.2) # find q for that pose
        q_target = q_target - q_adjust # adjust end effector orientation
        # check joint limits:
        if (q_target[6] > 2.8973):
            q_target[6] = q_target[6] - pi
        if (q_target[6] < -2.8973):
            q_target[6] = q_target[6] + pi
        arm.safe_move_to_position(q_target) # move arm to target q
        success = arm.exec_gripper_cmd(0.047, 25) # grab cube with 10 N force
        arm.safe_move_to_position(q_return) # move back to start position

        state = arm.get_gripper_state() 
        print("force reading: ")
        print(state["force"])
        print("position reading: ")
        print(state["position"])
        # # not using for now, but this is what we should use to check if we successfully picked up the block or not:
        # if state["force"][0] > 5:
        #     success = True
        # else:
        #     success = False
        return success, q_return

def grab_dynamic_block(seed,team):
    """
    Positions the arm to intercept blocks on the rotating table and attempts to grab one
    
    Returns:
    - success: bool indicating if a block was grabbed
    - seed: the configuration to return to
    """
    # Predefined pose for intercepting rotating blocks
    #dynamic_intercept_q = [-0.1, -0.3, 0.0, -1.8, 0.0, 1.5, 0.5]  
    #BLUE
    dynamic_shift = np.array([-0.3, 0, 0, 0, 0, 0, 0])
    #dynamic_intercept_q= np.array([ 0.6149055,   1.18500208,  0.6802492,  -0.91748964,  0.5906635,   1.77556375, -1.08117336]) + dynamic_shift
    #dynamic_intercept_q  = dynamic_red1 + np.array([-0.2, -0.5, 0, 0, 0, 0, 0])
    if team == 'red':
    	# RED
    	dynamic_intercept_q = np.array([-1.58490469+0.2, -1.25087031-.04,  2.31849405, -0.86902863,  1.87877124,  0.97045593+0.35, -1.22053536]) 
    	dynamic_mid = np.array([-1.58490469, -1.25087031-.04,  2.31849405, -0.86902863,  1.87877124,  0.97045593, -1.22053536]) + dynamic_shift
    if team == 'blue':
    
    	# BLUE
    	#dynamic_intercept_q = np.array([-1.58490469+pi+0.2, -1.25087031,  2.31849405, -0.86902863,  1.87877124,  0.97045593+0.35, -1.22053536]) 
    	dynamic_intercept_q = np.array([ 0.88516447, -1.58307014-0.06, -1.68525571, -0.93205714, -0.31681186,  1.85633407,
 -0.77035084])

    	#dynamic_mid = np.array([-1.58490469+pi, -1.25087031,  2.31849405, -0.86902863,  1.87877124,  0.97045593, -1.22053536]) + dynamic_shift
    	dynamic_mid = np.array([ 0.88516447-0.5, -1.58307014-0.06, -1.68525571, -0.93205714, -0.31681186,  1.85633407,
 -0.77035084-0.2])

    
    # Move to interception position
    arm.safe_move_to_position(dynamic_mid)
    open = arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH, 0) #open
    arm.safe_move_to_position(dynamic_intercept_q)
    
    # Wait for block to rotate into position (5 seconds)
    # print("Waiting for block rotation...")
    # time.sleep(5)
    
    # Try to grab whatever is there
    success = arm.exec_gripper_cmd(0.04, 35)
    
    # Return to safe position if grab was successful
    if success:
        arm.safe_move_to_position(seed)
        
    return success, seed

if __name__ == "__main__":
    try:
        team = rospy.get_param("team")
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()
    ik = IK()
    fk = FK()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 
                              0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position)

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
        #dynamic_intercept_q = np.array([-1.58490469+pi, -1.25087031,  2.31849405, -0.86902863,  1.87877124,  0.97045593+0.35, -1.22053536]) 
        # dynamic_mid_q = np.array([-1.58490469+pi, -1.25087031,  2.31849405, -0.86902863,  1.87877124,  0.97045593, -1.22053536]) + dynamic_shift
        #q_stat_red = ([-0.18066, -0.09311, -0.1158,  -1.73017, -0.02593,  1.75,  0.5])
        q_stat_red = ([ 8.22038400e-02, -8.71450024e-02,  2.21771336e-01, -1.71916998e+00, -2.79373130e-04,  1.63581915e+00,  1.09733342e+00])
        drop_off_H_dyn = np.array([[0, -1, 0, 0.542],
                          [-1, 0, 0, -0.26],
                          [0, 0, -1, 0.23],
                          [0, 0, 0, 1]])
        over_drop_off_H_dyn = np.array([[0, -1, 0, 0.542],
                               [-1, 0, 0, -0.26],
                               [0, 0, -1, 0.505],
                               [0, 0, 0, 1]])
        drop_off_H = np.array([[0, -1, 0, 0.652],
                           [-1, 0, 0, -0.10],
                           [0, 0, -1, 0.265],
                           [0, 0, 0, 1]])
        over_drop_off_H = np.array([[0, -1, 0, 0.652],
                           [-1, 0, 0, -0.10],
                           [0, 0, -1, 0.305],
                           [0, 0, 0, 1]])
    else:
        print("**  RED TEAM  **")
        #dynamic_intercept_q = dynamic_intercept_q_red
        #dynamic_mid_q = dynamic_mid_red
        q_stat_red = [-0.17154, -0.05208, -0.12823, -1.92441, -0.02569,  1.85,  0.5]
        drop_off_H_dyn = np.array([[0, -1, 0, 0.542],
                          [-1, 0, 0, 0.26],
                          [0, 0, -1, 0.23],
                          [0, 0, 0, 1]])
        over_drop_off_H_dyn = np.array([[0, -1, 0, 0.542],
                               [-1, 0, 0, 0.26],
                               [0, 0, -1, 0.305],
                               [0, 0, 0, 1]])
        drop_off_H = np.array([[0, -1, 0, 0.562],
                           [-1, 0, 0, 0.10],
                           [0, 0, -1, 0.265],
                           [0, 0, 0, 1]])
        over_drop_off_H = np.array([[0, -1, 0, 0.562],
                           [-1, 0, 0, 0.10],
                           [0, 0, -1, 0.305],
                           [0, 0, 0, 1]])
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")
    print("Go!\n")

    # Initial setup for dynamic block grabbing
    
    seed = q_stat_red.copy()
    
    # Initialize matrices for block placement
    #drop_off_H = np.array([[0, -1, 0, 0.562],
    #                      [-1, 0, 0, 0.18],
    #                      [0, 0, -1, 0.225],
    #                      [0, 0, 0, 1]])
    #over_drop_off_H = np.array([[0, -1, 0, 0.562],
    #                           [-1, 0, 0, 0.18],
    #                           [0, 0, -1, 0.305],
    #                           [0, 0, 0, 1]])
    
    shift_dynamic = np.array([[0, 0, 0, 0],
                       [0, 0, 0, -0.01],
                       [0, 0, 0, 0.01],
                       [0, 0, 0, 0]])
                       
    shift_static = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, -.015],
                       [0, 0, 0, 0]])
                       
    z_shift = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0.055],
                       [0, 0, 0, 0]])
    count = 0

    # Main control loop for dynamic blocks only
    start_time = time_in_seconds()
    drop_off_H_dyn += shift_dynamic
    while time_in_seconds() - start_time < 300:  # 5 minute time limit
    	 
        # Try to grab dynamic block
        over_drop_off, _, _, _ = ik.inverse(over_drop_off_H_dyn, seed, method='J_pseudo', alpha=.5)
        success, current_pos = grab_dynamic_block(over_drop_off,team)
        
        if success:
            #print("Successfully grabbed dynamic block!")
            # Use stacking logic for placement
            #over_drop_off_H = over_drop_off_H + z_shift*count
            #over_drop_off, _, _, _ = ik.inverse(over_drop_off_H, seed, method='J_pseudo', alpha=.5)
            #print("drop off q: ", over_drop_off)
            
            # Execute stacking sequence
            arm.safe_move_to_position(over_drop_off)
            drop_off_H_dyn = drop_off_H_dyn + z_shift * count
            drop_off, _, _, _ = ik.inverse(drop_off_H_dyn, over_drop_off, method='J_pseudo', alpha=.2)
            arm.safe_move_to_position(drop_off)
            drop = arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH, 0)
            
            # Return to ready position
            # over_drop_off, _, _, _ = ik.inverse(over_drop_off_H, over_drop_off, method='J_pseudo', alpha=.5)
            arm.safe_move_to_position(over_drop_off)
            #arm.safe_move_to_position(q_stat_red)
            count += 1
        # else:
        #     print("Failed to grab dynamic block, retrying...")
        #     # Small delay before next attempt
        #     time.sleep(1)
            
        if count > 1:
            break
            
            
    count = 0

    # get the transform from camera to panda_end_effector
    H_ee_camera = detector.get_H_ee_camera()
    _, T0e = fk.forward(q_stat_red)
    T0c = np.dot(T0e, H_ee_camera)
    print("T0c: ")
    print(T0c)
    bias_pose = np.array([[0, -1, 0, 0],
                          [-1, 0, 0, 0], 
                          [0, 0, -1, 0], 
                          [0, 0, 0, 1]])

    arm.safe_move_to_position(q_stat_red) # got to position on top of the cubes to see which one to pick up

    # Create queue for IK solutions
    pose_queue = Queue()
    threads = []

    # Start IK computation threads for all detected blocks
    detections = list(detector.get_detections())
    drop_off_H += shift_static
    for (block_name, pose) in detections:
        print("position of ", block_name, "in camera frame:")
        print(pose)
        pose_0 = np.dot(T0c, pose)  # change reference frame of pose from camera to base frame
        thread = IKThread(pose_0, seed, pose_queue)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Process all the pre-computed IK solutions
    while not pose_queue.empty():
        pose_0, q_over_target, target_adj, q_adjust = pose_queue.get()
        
        # Move to pre-computed position
        arm.safe_move_to_position(q_over_target)
        arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH, 0)
        
        # find pose 10 cm lower (center of cube):
        target_pose = target_adj - z_shift*2 

        # pick up block and return to safe position (q_stat_red):
        success, q_current = pick_up_target(target_pose, q_over_target, q_adjust, q_stat_red)

        if success:
            print("Cube was grabbed: ", success)
            over_drop_off_H = over_drop_off_H + z_shift*count
            over_drop_off, _, _, _ = ik.inverse(over_drop_off_H, seed, method='J_pseudo', alpha=.5)
            print("drop off q: ")
            print(over_drop_off)
            arm.safe_move_to_position(over_drop_off)
            drop_off_H = drop_off_H + z_shift*count
            drop_off, _, _, _ = ik.inverse(drop_off_H, over_drop_off, method='J_pseudo', alpha=.2)
            arm.safe_move_to_position(drop_off)
            drop = arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH, 0)
            over_drop_off, _, _, _ = ik.inverse(over_drop_off_H, over_drop_off, method='J_pseudo', alpha=.5)
            arm.safe_move_to_position(over_drop_off)
            arm.safe_move_to_position(q_stat_red)
            count = 1
        else:
            print("Cube was grabbed: ", success)
            
            
            
"""
HOW THIS CODE WORKS - A Complete Explanation

This code implements a robotic pick-and-place system for grabbing blocks from a rotating table
and stacking them. Here's how it works step by step:

1. INITIALIZATION AND SETUP
   - Creates robot arm controller, object detector, and kinematics solvers
   - Sets up initial positions and transformation matrices for block placement
   - Defines the target platform location where blocks will be stacked

2. DYNAMIC BLOCK GRABBING STRATEGY
   The grab_dynamic_block() function:
   - Moves the robot arm to a predefined position (dynamic_intercept_q) near the rotating table
   - Waits 5 seconds for a block to rotate into the gripper's position
   - Attempts to grab any block that enters the gripper zone
   - Returns to a safe position if successful

3. BLOCK STACKING PROCESS
   When a block is successfully grabbed, the code:
   - Calculates the next stacking position based on how many blocks are already stacked (count)
   - Moves to a position above the stack (over_drop_off_H)
   - Lowers the block into position (drop_off_H)
   - Releases the block
   - Returns to the ready position
   - Increments the stack counter

4. CONTINUOUS OPERATION
   - Runs for 5 minutes (300 seconds)
   - Continuously attempts to grab blocks from the rotating table
   - If a grab fails, waits 1 second before trying again
   - Keeps track of successful grabs and adjusts stack height accordingly

5. KEY PARAMETERS:
   - dynamic_intercept_q: Robot arm position for intercepting rotating blocks
   - seed: Reference configuration for inverse kinematics
   - drop_off_H: Target position for placing blocks
   - z_shift: Height increment for stacking blocks
   - count: Tracks number of successfully stacked blocks

6. SAFETY FEATURES:
   - Uses safe_move_to_position for collision-free motion
   - Returns to safe positions after operations
   - Implements proper gripper control

To use this code:
1. Make sure the robot is properly initialized
2. Tune dynamic_intercept_q for your specific table setup
3. Run the code and press ENTER to begin
4. The robot will automatically attempt to grab and stack blocks for 5 minutes

Note: This code is specifically designed for grabbing blocks from a rotating table
and does not handle static blocks. The success of grabbing depends heavily on:
- The correct positioning of dynamic_intercept_q
- The rotation speed of the table
- The timing of the grab attempts (5-second wait)
"""
