"""
Module biến đổi hệ tọa độ đa cảm biến về hệ quy chiếu thân xe (Ego-Vehicle)
"""
import numpy as np
import torch
from pyquaternion import Quaternion

def quaternion_to_matrix(q):
    """Chuyển quaternion [w, x, y, z] sang ma trận quay 3x3"""
    if isinstance(q, list):
        q = Quaternion(q)
    return q.rotation_matrix

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
