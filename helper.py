import numpy as np
import math

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


# Determine whether the matrix R is a valid rotation matrix
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculate the eular angles from a pose matrix (3x4)
def R_to_angle(Rt):

    Rt = np.reshape(np.array(Rt), (3,4))
    t = Rt[:,-1]
    R = Rt[:,:3]

    assert(isRotationMatrix(R))
    
    x, y, z = euler_from_matrix_new(R)
    theta = [x, y, z]
    pose_15 = np.concatenate((theta, t, R.flatten()))
    assert(pose_15.shape == (15,))
    return pose_15

# Calculate the relative rotation (Eular angles) and translation from two pose matrices
# Rt1: a list of 12 floats
# Rt2: a list of 12 floats
def cal_rel_pose(Rt1, Rt2):
    Rt1 = np.reshape(np.array(Rt1), (3,4))
    Rt1 = np.concatenate((Rt1, np.array([[0,0,0,1]])), 0)
    Rt2 = np.reshape(np.array(Rt2), (3,4))
    Rt2 = np.concatenate((Rt2, np.array([[0,0,0,1]])), 0)
    
    # Calculate the relative transformation Rt_rel
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2    

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]
    assert(isRotationMatrix(R_rel))
    
    # Extract the Eular angle from the relative rotation matrix
    x, y, z = euler_from_matrix_new(R_rel)
    theta = [x, y, z]
    
    rod = SO3_to_so3(R_rel)
    
    pose_rel_9 = np.concatenate((theta, t_rel, rod))

    assert(pose_rel_9.shape == (9,))
    return pose_rel_9

# Calculate the 3x4 transformation matrix from Eular angles and translation vector
# pose: (3 angles, 3 translations) 
def angle_to_R(pose):
    R = eulerAnglesToRotationMatrix(pose[:3])
    t = pose[3:].reshape(3, 1)    
    R = np.concatenate((R, t), 1)
    return R


# Calculate the rotation matrix from eular angles (roll, yaw, pitch)
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

# Calculate the eular angles (roll, yaw, pitch) from a rotation matrix
def euler_from_matrix_new(matrix):
    
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    
    cy = math.sqrt(M[0, 0]*M[0, 0] + M[1, 0]*M[1, 0])
    ay = math.atan2(-M[2, 0], cy)

    if ay < -math.pi/2 + _EPS and ay > -math.pi/2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2( -M[1, 2],  -M[0, 2])
    elif ay < math.pi/2 + _EPS and ay > math.pi/2 - _EPS:
        ax = 0
        az = math.atan2( M[1, 2],  M[0, 2])
    else:
        ax = math.atan2( M[2, 1],  M[2, 2])
        az = math.atan2( M[1, 0],  M[0, 0])

    return np.array([ax, ay, az])

# Normalization angles to constrain that it is between -pi and pi   
def normalize_angle_delta(angle):
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

# Transformation from SO3 to so3
def SO3_to_so3(R):
    assert(isRotationMatrix(R))
    cos = ((R[0,0]+R[1,1]+R[2,2])-1)/2
    theta = np.arccos(max(min(cos,1.0),-1.0))
    logR = (theta/np.sin(theta)/2)*(R-R.T)
    w = np.array([logR[2,1], logR[0,2], logR[1,0]])
    return w

# Transformation from so3 to SO3
def so3_to_SO3(w):
    theta = np.sqrt(np.sum(w**2))
    w_hat = np.array([ [    0, -w[2],   w[1] ],
                       [ w[2],     0,  -w[0] ],
                       [-w[1],  w[0],     0  ]
                     ])
    w_hat_2 = w_hat @ w_hat
    R = np.eye(3) + (np.sin(theta)/theta)*w_hat + ((1-np.cos(theta))/theta**2)*w_hat_2
    return R

# Transformation from SE3 to se3
def SE3_to_se3(Rt):

    Rt = np.reshape(np.array(Rt), (3,4))
    t = Rt[:,-1]
    R = Rt[:,:3]
    assert(isRotationMatrix(R))

    cos = ((R[0,0]+R[1,1]+R[2,2])-1)/2
    theta = np.arccos(max(min(cos,1.0),-1.0))
    logR =  (theta/np.sin(theta)/2)*(R-R.T)

    w = [logR[2,1], logR[0,2], logR[1,0]]
    
    alpha = 1 - (theta*np.cos(theta/2))/(2*np.sin(theta/2))
    V_inv = np.eye(3) - 0.5*logR + (alpha/theta**2)*(logR @ logR)
    t_ = V_inv.dot(t)

    v = np.concatenate((t_, w))
    return v

# Transformation from se3 to SE3
def se3_to_SE3(v):

    t_ = v[:3]
    w = v[3:]

    theta = np.sqrt(np.sum(w**2))
    w_hat = np.array([ [    0, -w[2],   w[1] ],
                       [ w[2],     0,  -w[0] ],
                       [-w[1],  w[0],     0  ]
                     ])
    
    w_hat_2 = w_hat @ w_hat
    R = np.eye(3) + (np.sin(theta)/theta)*w_hat + ((1-np.cos(theta))/theta**2)*w_hat_2    
    V = np.eye(3) + ((1-np.cos(theta))/theta**2)*w_hat + ((theta-np.sin(theta))/theta**3)*w_hat_2
    t = V.dot(t_).reshape(3,1)

    R = np.concatenate((R, t), 1)
    return R




if __name__ == '__main__':
    #clean_unused_images()
    pose1 = np.array([0.3, 0.4, 0.5, 0, 2, 3])
    pose2 = np.array([0.5, 0.1, 0.5, 1, 2, 2])
    
    Rt1 = angle_to_R(pose1)
    pose1_ = R_to_angle(Rt1.flatten().tolist())

    Rt2 = angle_to_R(pose2)

    R1, t1 = Rt1[:,:3], Rt1[:, 3]
    R2, t2 = Rt2[:,:3], Rt2[:, 3]

    R3 = R1 @ R2
    t3 = R1.dot(t2) + t1
    Rt3 = np.concatenate((R3, t3.reshape(3,1)), 1)

    v1 = SE3_to_se3(Rt1.flatten().tolist())
    Rt1_ = se3_to_SE3(v1)

    v2 = SE3_to_se3(Rt2.flatten().tolist())
    Rt2_ = se3_to_SE3(v2)

    v3 = v1+v2
    v3_ = SE3_to_se3(Rt3.flatten().tolist())
    Rt3_ = se3_to_SE3(v3)
    
    pose3 = R_to_angle(Rt3.flatten().tolist())
    pose3_ = R_to_angle(Rt3_.flatten().tolist())
    

    pose = cal_rel_pose(Rt1.flatten().tolist(), Rt3.flatten().tolist())
    import pdb; pdb.set_trace()  # breakpoint c322e71c //