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
    if q is None:
        return np.eye(3, dtype=np.float32)
    if HAVE_PYQUAT:
        try:
            if isinstance(q, (list, np.ndarray, tuple)):
                q = Quaternion(q)
            return q.rotation_matrix.astype(np.float32)
        except Exception:
            pass
    # Fallback thuần toán học NumPy
    try:
        if len(q) == 4:
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            return np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
                [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
            ], dtype=np.float32)
    except Exception:
        pass
    return np.eye(3, dtype=np.float32)

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

def _extract_pose(pose_dict):
    """Trích xuất an toàn (rotation, translation) từ nhiều cấu trúc dictionary khác nhau"""
    if not isinstance(pose_dict, dict):
        return [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    
    # Nếu bị bọc trong dict con 'ego_pose'
    if 'ego_pose' in pose_dict and isinstance(pose_dict['ego_pose'], dict):
        pose_dict = pose_dict['ego_pose']

    rot = (
        pose_dict.get('ego2global_rotation') or 
        pose_dict.get('rotation') or 
        pose_dict.get('rot') or 
        [1.0, 0.0, 0.0, 0.0]
    )
    trans = (
        pose_dict.get('ego2global_translation') or 
        pose_dict.get('translation') or 
        pose_dict.get('trans') or 
        [0.0, 0.0, 0.0]
    )
    return rot, trans

def compute_ego_relative_transform(ego_pose_prev, ego_pose_curr):
    """
    Tính ma trận biến đổi 4x4 từ hệ quy chiếu Ego frame t-1 sang Ego frame t
    T_(t-1 -> t) = (T_curr)^(-1) * T_prev
    """
    try:
        rot_prev, trans_prev = _extract_pose(ego_pose_prev)
        rot_curr, trans_curr = _extract_pose(ego_pose_curr)

        R_prev = quaternion_to_matrix(rot_prev)
        t_prev = np.array(trans_prev, dtype=np.float32).reshape(3, 1)
        T_prev = np.eye(4, dtype=np.float32)
        T_prev[:3, :3] = R_prev
        T_prev[:3, 3:] = t_prev

        R_curr = quaternion_to_matrix(rot_curr)
        t_curr = np.array(trans_curr, dtype=np.float32).reshape(3, 1)
        T_curr = np.eye(4, dtype=np.float32)
        T_curr[:3, :3] = R_curr
        T_curr[:3, 3:] = t_curr

        T_rel = np.dot(np.linalg.inv(T_curr), T_prev)
        return torch.tensor(T_rel, dtype=torch.float32)
    except Exception:
        return torch.eye(4, dtype=torch.float32)
