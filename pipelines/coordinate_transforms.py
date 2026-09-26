import numpy as np
import torch

try:
    from pyquaternion import Quaternion
    HAVE_PYQUAT = True
except ImportError:
    HAVE_PYQUAT = False

def quaternion_to_matrix(q):
    """
    Chuyển quaternion [w, x, y, z] sang ma trận quay 3x3.
    Có sẵn fallback thuần NumPy nếu máy chưa cài pyquaternion.
    """
    if HAVE_PYQUAT:
        if isinstance(q, (list, np.ndarray)):
            q = Quaternion(q)
        return q.rotation_matrix
    else:
        # Fallback thuần toán học NumPy
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)

def transform_points(points, translation, rotation):
    """
    Biến đổi tập điểm 3D [N, 3+] bằng translation [x, y, z] và rotation (quaternion hoặc ma trận 3x3)
    """
    pts_xyz = points[:, :3]
    if isinstance(rotation, (list, tuple)):
        rot_mat = quaternion_to_matrix(rotation)
    else:
        rot_mat = np.array(rotation)
    
    trans_vec = np.array(translation).reshape(1, 3)
    transformed_xyz = np.dot(pts_xyz, rot_mat.T) + trans_vec
    
    out_points = points.copy()
    out_points[:, :3] = transformed_xyz
    return out_points

def compute_ego_relative_transform(ego_pose_prev, ego_pose_curr):
    """
    Tính ma trận biến đổi 4x4 từ hệ quy chiếu Ego frame t-1 sang Ego frame t
    T_(t-1 -> t) = (T_curr)^(-1) * T_prev
    """
    R_prev = quaternion_to_matrix(ego_pose_prev['rotation'])
    t_prev = np.array(ego_pose_prev['translation']).reshape(3, 1)
    T_prev = np.eye(4)
    T_prev[:3, :3] = R_prev
    T_prev[:3, 3:] = t_prev

    R_curr = quaternion_to_matrix(ego_pose_curr['rotation'])
    t_curr = np.array(ego_pose_curr['translation']).reshape(3, 1)
    T_curr = np.eye(4)
    T_curr[:3, :3] = R_curr
    T_curr[:3, 3:] = t_curr

    T_rel = np.dot(np.linalg.inv(T_curr), T_prev)
    return torch.tensor(T_rel, dtype=torch.float32)
