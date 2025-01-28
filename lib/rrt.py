import numpy as np
from lib.detectCollision import detectCollision
from lib.calculateFK import FK
from lib.loadmap import loadmap
from copy import deepcopy

class Node:
    def __init__(self, config, parent_index=None):
        self.config = np.array(config)  # The joint configuration
        self.parent_index = parent_index  # Index of the parent node in the tree

def find_nearest_node(tree, q_rand):
    """
    Find the nearest node in the tree to the random configuration q_rand.
    """
    distances = [np.linalg.norm(node.config - q_rand) for node in tree]
    nearest_index = np.argmin(distances)
    return nearest_index

def is_within_limits(position, lower_lim, upper_lim):
    return np.all(position >= lower_lim) and np.all(position <= upper_lim)

def is_path_collision_free(q1, q2, obstacles, steps=10):
    path_points = np.linspace(q1, q2, steps+1)
    # We have to convert from 7D to a 3D space
    
    configs = np.linspace(q1, q2, steps)
    FK_instance = FK()
    joint_positions1,_ = FK_instance.forward(q1)
    joint_positions2,_ = FK_instance.forward(q2)
    
    for obstacle in obstacles:
        # detectCollision returns true if there exists a collision
        # so if any of the elements of the boolean array are true then there exists a collision
        if any(detectCollision(joint_positions1, joint_positions2, obstacle)):
            return False
    return True

def rrt(map, start, goal):
    """
    Implement bidirectional RRT algorithm for the Panda arm.
    :param map: the map struct
    :param start: start pose of the robot (1x7 numpy array)
    :param goal: goal pose of the robot (1x7 numpy array)
    :return: returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
             the path. The first row is start and the last row is goal. If no path is found, returns an empty array.
    """
    lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upperLim = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    
    tree_start = [Node(start)]
    tree_goal = [Node(goal)]
    
    maxIter = 1000
    step_size = 0.9

    if not (is_within_limits(start, lowerLim, upperLim) and is_path_collision_free(start,start, map.obstacles)):
        print("Start position is invalid or in collision.")
        return np.array([])
    if not (is_within_limits(goal, lowerLim, upperLim) and is_path_collision_free(goal,goal, map.obstacles)):
        print("Goal position is invalid or in collision.")
        return np.array([])

    for i in range(maxIter):
        if i % 2 == 0:
            tree_a, tree_b = tree_start, tree_goal
        else:
            tree_a, tree_b = tree_goal, tree_start

        rand_prob = np.random.rand()  # Generate a random number between 0 and 1
        if rand_prob > 0.15:
            q_rand = np.random.uniform(lowerLim, upperLim)  # Random configuration within limits
        else:
            # If probability is <= 0.15, use goal or start depending on the iteration
            if i % 2 == 0:
                q_rand = goal  # Use goal if i is even
            else:
                q_rand = start  # Use start if i is odd
        
        # Find nearest node in tree_a
        nearest_index = find_nearest_node(tree_a, q_rand)
        q_nearest = tree_a[nearest_index].config

        # Extend towards q_rand
        direction = q_rand - q_nearest
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.zeros_like(direction)
        q_new = q_nearest + direction * step_size
        
        # Check if q_new is valid
        if (np.all(q_new >= lowerLim) and np.all(q_new <= upperLim) and
                is_path_collision_free(q_nearest, q_new, map.obstacles)):
            tree_a.append(Node(q_new, nearest_index))

            # Try to connect to tree_b
            nearest_index_b = find_nearest_node(tree_b, q_new)
            q_nearest_b = tree_b[nearest_index_b].config
            
            if is_path_collision_free(q_new, q_nearest_b, map.obstacles):
                # Path found
                if i % 2 == 0:
                    path_start = extract_path(tree_start, len(tree_a) - 1)
                    path_goal = extract_path(tree_goal, nearest_index_b)
                    return np.vstack((path_start, path_goal[::-1]))
                else:
                    path_goal = extract_path(tree_goal, len(tree_a) - 1)
                    path_start = extract_path(tree_start, nearest_index_b)
                    return np.vstack((path_start, path_goal[::-1]))
    print("Failed to find a path.")
    return np.array([])

def extract_path(tree, end_index):
    # Extracting a path from a tree
    path = []
    current_index = end_index
    while current_index is not None:
        path.append(tree[current_index].config)
        current_index = tree[current_index].parent_index
    return np.array(path[::-1])  # Reverse to get the path from start to end

if __name__ == '__main__':
    map_struct = loadmap("/Users/neelmulay/Documents/Penn/MEAM 5200/Labs/lib/maps/map4.txt")
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    print(path)