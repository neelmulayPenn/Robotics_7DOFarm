# Robotics 7DOF Arm: Full Pick-and-Place Stack

This repository contains a complete kinematics and planning stack for a 7-DOF Franka Panda arm, including:

- Forward kinematics (FK)
- Geometric Jacobian
- Velocity-level and position-level inverse kinematics (IK)
- Manipulability analysis
- Collision checking against axis-aligned obstacles
- Joint-space planning with Potential Fields and bidirectional RRT
- Pick-and-place orchestration for static and dynamic blocks

The core implementation is in `lib/`.

[![Pick and Place Competition](https://youtu.be/bMnCxwn1p1U)](https://youtu.be/bMnCxwn1p1U)

## Project Highlights

- Built a gradient-descent IK solver with null-space joint-centering.
- Implemented velocity IK and velocity IK with secondary null-space tasks.
- Developed and evaluated two planners:
  - Local planner: artificial potential fields
  - Global planner: bidirectional RRT with goal biasing
- Integrated planning + IK + gripper control into a full pick-and-place pipeline.
- Added dynamic block interception logic for rotating-table tasks.


## Full Pick-and-Place Stack

The manipulation pipeline in `lib/final.py` is structured as a practical task loop that combines:

- Perception-driven target selection (`ObjectDetector`)
- Pose shaping and approach planning (`fix_pose`)
- Iterative IK solves (`IK.inverse`)
- Safe motion execution (`arm.safe_move_to_position`)
- Grasp/release control with force commands (`arm.exec_gripper_cmd`)

### How The Stack Works (End-to-End)

1. Acquire current robot state and candidate block pose from perception.
2. Transform the raw block pose into a grasp-friendly pose using `fix_pose(...)`:
   - Apply a tool orientation bias so the gripper is aligned for top-down grasping.
   - Add a z-offset to create an approach pose above the block.
3. Run IK to approach from above (`J_pseudo`, larger alpha for faster convergence).
4. Move to the approach pose and open gripper.
5. Run IK again for the actual grasp depth (`J_pseudo`, smaller alpha for tighter convergence).
6. Descend and close gripper with force target.
7. Lift/retreat to a safer return configuration.
8. Solve IK for drop-off approach and place pose.
9. Move to drop-off, release, and reset for next cycle.

### Core Strategy: Static Blocks

Static blocks are solved as a precision pipeline:

- Goal source: direct vision pose estimate.
- Approach policy: move to a pre-grasp point above the block before descending.
- Orientation policy: compensate end-effector yaw using `q_adjust` so finger direction matches block orientation.
- Control emphasis: accuracy over speed (more deliberate descend and place sequence).
- Reliability behavior: grasp success is checked via gripper command/state and robot retreats before the next pick.

This strategy is robust for table-top blocks where target motion is negligible and geometric alignment is the dominant challenge.

### Core Strategy: Dynamic Blocks (Rotating Turntable)

Dynamic blocks are solved as an interception problem:

- Goal source: known dynamic capture region rather than repeated full re-planning to moving targets.
- Interception policy: move first to `dynamic_mid`, then to `dynamic_intercept_q` timed for block arrival.
- Grasp policy: keep gripper open during approach, then close with higher force at intercept.
- Timing policy: prioritize synchronized capture over perfect pose matching.
- Recovery policy: once grasp succeeds, immediately return to a safe seed pose and continue to drop-off.

This trades geometric optimality for temporal robustness, which is appropriate when the object is moving on a predictable path.

### State-Machine View

You can think of the implementation as this state machine:

`SEARCH -> PREGRASP -> GRASP -> RETREAT -> PLACE -> RESET`

with a dynamic branch:

`SEARCH_DYNAMIC -> INTERCEPT_MID -> INTERCEPT_GRAB -> RETREAT -> PLACE -> RESET`

### Pick-and-Place Code Snippet (Static Path)

```python
target_adj, yaw = fix_pose(target)
q_adjust = np.array([0, 0, 0, 0, 0, 0, yaw])

# Pre-grasp above object
q_over_target, _, _, _ = ik.inverse(target_adj, current, method='J_pseudo', alpha=.5)
q_over_target = q_over_target - q_adjust
arm.safe_move_to_position(q_over_target)
arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH, 0)

# Descend and grasp
q_target, _, _, _ = ik.inverse(target, q_over_target, method='J_pseudo', alpha=.2)
q_target = q_target - q_adjust
arm.safe_move_to_position(q_target)
success = arm.exec_gripper_cmd(0.047, 25)
arm.safe_move_to_position(q_over_target)
```

### Dynamic Interception Snippet

```python
arm.safe_move_to_position(dynamic_mid)
arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH, 0)  # open
arm.safe_move_to_position(dynamic_intercept_q)

# Time-critical close at intercept pose
success = arm.exec_gripper_cmd(0.04, 35)
if success:
	arm.safe_move_to_position(seed)
```

### Why This Works Well In Practice

- Separating pre-grasp and grasp reduces collision risk and IK instability.
- Re-using warm-start joint seeds improves solve speed.
- The dynamic mode avoids expensive continuous re-planning and instead exploits predictable object motion.
- Shared drop-off logic keeps the post-grasp behavior consistent between static and dynamic tasks.

## Kinematics Stack

### 1) Forward Kinematics (FK)

File: `lib/calculateFK.py`

- Uses DH parameters for the Franka Panda.
- Returns both joint-center positions and end-effector transform.
- Output:
  - `jointPositions`: 8x3 (base + 7 joints/end frame)
  - `T0e`: 4x4 homogeneous transform

Technical details:

- Each link transform is constructed as a standard DH matrix

	```math
	A_i = \begin{bmatrix}
	c_{\theta_i} & -s_{\theta_i}c_{\alpha_i} & s_{\theta_i}s_{\alpha_i} & a_i c_{\theta_i} \\
	s_{\theta_i} & c_{\theta_i}c_{\alpha_i} & -c_{\theta_i}s_{\alpha_i} & a_i s_{\theta_i} \\
	0 & s_{\alpha_i} & c_{\alpha_i} & d_i \\
	0 & 0 & 0 & 1
	\end{bmatrix}
	```

- The full end-effector pose is the chained product $T_{0e} = A_1 A_2 \cdots A_7$.
- The implementation applies per-joint geometric offsets (`offset_joint_positions`) after each cumulative transform so reported joint centers match the physical robot geometry.
- This FK output is reused downstream by Jacobian, IK validation, and collision checking.

#### FK Snippet

```python
def forward(self, q):
	jointPositions = np.zeros((8,3))
	T0e = np.identity(4)
	jointPositions[0] = [0, 0, 0.141]

	for i in range(len(self.dh_params)):
		a, alpha, d, theta_offset = self.dh_params[i]
		A = self.dh_transform(a, alpha, d, q[i] + theta_offset)
		T0e = T0e @ A
		joint_pos = T0e @ self.offset_joint_positions[i+1].reshape(4, 1)
		jointPositions[i+1] = joint_pos[:3, 0]

	return jointPositions, T0e
```

### 2) Geometric Jacobian

File: `lib/calcJacobian.py`

- Computes a 6x7 Jacobian in world frame.
- Top 3 rows: linear velocity mapping.
- Bottom 3 rows: angular velocity mapping.

Technical details:

- The Jacobian is built column-wise from current frame axes and origins:

```math
J_v^{(i)} = z_{i-1} \times (o_n - p_{i-1}), \quad J_\omega^{(i)} = z_{i-1}
```

- `z_prev` and `p_prev` are updated through the same forward chain used for FK, keeping the Jacobian exactly consistent with your kinematic model.
- Output twist relation is

```math
\begin{bmatrix} v \\ \omega \end{bmatrix} = J(q)\dot{q}
```

#### Jacobian Snippet

```python
J_v = np.cross(z_prev, o_n - p_prev)
J_w = z_prev
J[:3, i] = J_v
J[3:, i] = J_w
```

### 3) Forward Velocity Kinematics

File: `lib/FK_velocity.py`

- Computes end-effector twist from joint velocity:

```python
J = calcJacobian(q_in)
velocity = J @ dq
```

## Inverse Kinematics

### 1) Velocity IK

File: `lib/IK_velocity.py`

- Solves least-squares IK at velocity level.
- Supports unconstrained Cartesian components via `NaN` masking.

Technical details:

- Cartesian command is stacked as $v_{task} = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]^T$.
- If any task component is unconstrained, that row is removed from both $J$ and $v_{task}$.
- The solver computes the minimum-residual least-squares solution:

```math
\dot{q} = \arg\min_{\dot{q}} \lVert J_r \dot{q} - v_r \rVert_2^2
```

- Implemented with `np.linalg.lstsq`, which is numerically stable near singular configurations compared to directly inverting normal equations.

#### Velocity IK Snippet

```python
vel_target = np.vstack((v_in.reshape((3,1)), omega_in.reshape((3,1))))
mask = ~np.isnan(vel_target)
J_reduced = J[mask.flatten(), :]
vel_reduced = vel_target[mask]
dq, _, _, _ = np.linalg.lstsq(J_reduced, vel_reduced, rcond=None)
```

### 2) Velocity IK with Null-Space Task

File: `lib/IK_velocity_null.py`

- Adds a secondary joint-space objective `b` projected through Jacobian null space.

Technical details:

- Primary task is solved with pseudo-inverse.
- Secondary objective is projected with

```math
N = I - J^+J
```

- Final command:

```math
\dot{q} = J^+ v_r + N b
```

- This preserves the primary end-effector velocity while using residual DoF for posture shaping (e.g., joint-limit avoidance).

#### Velocity IK Null-Space Snippet

```python
J_pseudoinv = np.linalg.pinv(J_reduced)
dq = np.dot(J_pseudoinv, vel_reduced).squeeze()
null_space_proj = np.identity(7) - np.dot(J_pseudoinv, J_reduced)
null = np.dot(null_space_proj, b).squeeze()
return dq + null
```

### 3) Position IK (Gradient Descent + Null-Space)

File: `lib/IK_position_null.py`

- Iterative IK solver for full pose (position + orientation).
- Primary task: reduce pose error.
- Secondary task: center joints within limits.
- Supports both Jacobian pseudo-inverse and Jacobian transpose updates.

Technical details:

- Pose error is formed from translation and orientation:
  - Translation term from end-effector origin difference.
  - Orientation term from relative rotation skew-axis (axis-angle proxy).
- Primary update is either

```math
\Delta q_{ik} = -J^+ e \quad \text{or} \quad \Delta q_{ik} = -J^T e
```

- Secondary centering objective uses normalized joint offsets inside limits:

```math
\Delta q_{center} = -k\,\frac{2(q-q_{mid})}{q_{max}-q_{min}}
```

- Null-space blending:

```math
\Delta q = \Delta q_{ik} + (I - J^+J)\Delta q_{center}
```

- Iteration rule: $q_{k+1} = q_k + \alpha\Delta q$.
- Termination and validation are explicit in code:
  - Max iteration cap.
  - Minimum step-size convergence check.
  - Final pose tolerance check (linear + angular).
  - Hard joint-limit feasibility check.

#### Position IK Snippet

```python
dq_ik = IK.end_effector_task(q, target, method)
dq_center = IK.joint_centering_task(q)
J = calcJacobian(q)
J_pseudoinv = np.linalg.pinv(J)
nullspace_proj = np.identity(7) - np.dot(J_pseudoinv, J)
dq = dq_ik + np.dot(nullspace_proj, dq_center)
q += alpha * dq
```

## Planning Stack

### 1) Potential Field Planner

File: `lib/potentialFieldPlanner.py`

- Plans in joint space using attractive and repulsive forces.
- Computes virtual forces in Cartesian space for all joints/end-effector.
- Maps forces to joint torques using Jacobian transpose.
- Includes random-walk escape when in collision/minima.

Technical details:

- Attractive field uses distance-based switching (conic far from goal, parabolic near goal).
- Repulsive force is active only within a finite influence radius around obstacles.
- Per-joint force accumulation:

```math
F_i = F_{att,i} + \sum_j F_{rep,i}^{(j)}
```

- Force-to-torque mapping uses per-point Jacobians from `calculateFKJac.py`:

```math
tau = \sum_i J_{v,i}^T F_i
```

- Direction update is normalized and biased by a goal-seeking joint-space term:

```math
\Delta q \propto \tau - 5(q-q_{goal})
```

- The planner performs interpolation-based collision checks between successive joint configurations and triggers random-walk perturbations if stuck in local minima or collision.

#### Potential Fields Gradient Snippet

```python
joint_forces = PotentialFieldPlanner.compute_forces(
	joint_position_target, obstacle, joint_position_curr
)
joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)
joint_torques = joint_torques[:, :7]
joint_torques_magic = joint_torques - 5 * (q - target)
dq = joint_torques_magic / np.linalg.norm(joint_torques_magic)
```

### 2) Bidirectional RRT

File: `lib/rrt.py`

- Two trees grow from start and goal.
- Alternates expansion direction each iteration.
- Uses goal bias and fixed step expansion.
- Connects trees when collision-free bridge is found.

Technical details:

- State space is full 7-DoF joint space with hard limits.
- Sampling policy mixes global exploration with exploitation:
  - 85% uniform random samples.
  - 15% direct bias to opposite root (goal/start depending on active tree).
- Steering step:

```math
q_{new} = q_{near} + \eta\,\frac{q_{rand}-q_{near}}{\lVert q_{rand}-q_{near}\rVert}
```

where `eta = 0.9` in your implementation.

- Edge validity is checked with interpolated collision tests before node insertion.
- Once a new node can connect collision-free to the opposite tree, parent pointers are backtracked and concatenated into a start-to-goal path.
- Time/performance behavior:
  - Strong at escaping local minima where potential fields can stall.
  - Path quality is feasible-first (not optimal), with no post-smoothing stage yet.

#### RRT Snippet

```python
nearest_index = find_nearest_node(tree_a, q_rand)
q_nearest = tree_a[nearest_index].config
direction = q_rand - q_nearest
direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-6 else np.zeros_like(direction)
q_new = q_nearest + direction * step_size

if is_path_collision_free(q_nearest, q_new, map.obstacles):
	tree_a.append(Node(q_new, nearest_index))
```

## Collision + Environment Modeling

### Collision Detection

File: `lib/detectCollision.py`

- Uses line-segment vs axis-aligned-box intersection.
- Checks robot links (segment pairs) against each obstacle block.

```python
if any(detectCollision(joint_positions1, joint_positions2, obstacle)):
	return False
```

### Map Loading

File: `lib/loadmap.py`

- Parses map text files and extracts obstacle blocks into `map_struct.obstacles`.

```python
if words[0] == "block":
	obstacles = np.append(obstacles, np.array([[float(words[i]) for i in range(1, len(words))]]), axis=0)
```

## Manipulability Analysis

File: `lib/calcManipulability.py`

- Computes linear manipulability matrix:

```python
M = J_pos @ J_pos.T
```

- Computes manipulability index from singular values:

```python
singular_values = np.linalg.svd(J_pos, compute_uv=False)
mu = np.prod(singular_values)
```

## How to Use This Stack

### Minimal FK / Jacobian / IK Example

```python
import numpy as np
from lib.calculateFK import FK
from lib.calcJacobian import calcJacobian
from lib.IK_position_null import IK

fk = FK()
ik = IK()

q_seed = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
joint_pos, T0e = fk.forward(q_seed)
J = calcJacobian(q_seed)

target = T0e.copy()
target[0, 3] += 0.05  # move 5 cm in x
q_sol, rollout, success, msg = ik.inverse(target, q_seed, method='J_pseudo', alpha=0.25)
```

### Planner Example (RRT)

```python
import numpy as np
from copy import deepcopy
from lib.loadmap import loadmap
from lib.rrt import rrt

map_struct = loadmap("../maps/map1.txt")
start = np.array([0, -1, 0, -2, 0, 1.57, 0])
goal = np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
```

### Planner Example (Potential Fields)

```python
from lib.potentialFieldPlanner import PotentialFieldPlanner

planner = PotentialFieldPlanner(tol=1e-2, max_steps=2000, min_step_size=1e-5)
path = planner.plan(map_struct, start, goal)
```

