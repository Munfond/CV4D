"""
Hệ thống Đo lường & Đánh giá (Metrics Module)
Theo chuẩn bài báo Survey (CVPR 2024 / OccScore)
- Voxel IoU (hình học)
- mIoU (ngữ nghĩa)
- mAVE (sai số vận tốc Occupancy Flow)
- OccScore (Equation 18)
"""
import numpy as np
import torch
from configs.base_config import Config

class OccupancyMetrics:
    def __init__(self, num_classes=Config.NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # Ma trận nhầm lẫn (Confusion Matrix)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.velocity_errors = []

    def update(self, occ_pred, occ_gt, flow_pred=None, flow_gt=None):
        """
        occ_pred: [Z, Y, X] (chứa nhãn từ 0 -> 17)
        occ_gt: [Z, Y, X]
        flow_pred: [3, Z, Y, X]
        flow_gt: [3, Z, Y, X]
        """
        if torch.is_tensor(occ_pred):
            occ_pred = occ_pred.detach().cpu().numpy()
        if torch.is_tensor(occ_gt):
            occ_gt = occ_gt.detach().cpu().numpy()

        pred_flat = occ_pred.flatten()
        gt_flat = occ_gt.flatten()

        # Cập nhật confusion matrix
        mask = (gt_flat >= 0) & (gt_flat < self.num_classes)
        self.confusion_matrix += np.bincount(
            self.num_classes * gt_flat[mask] + pred_flat[mask],
            minlength=self.num_classes ** 2
        ).reshape((self.num_classes, self.num_classes))

        # Tính sai số vận tốc mAVE cho các voxel di động (classes 2->10: xe, người, etc.)
        if flow_pred is not None and flow_gt is not None:
            if torch.is_tensor(flow_pred):
                flow_pred = flow_pred.detach().cpu().numpy()
            if torch.is_tensor(flow_gt):
                flow_gt = flow_gt.detach().cpu().numpy()

            # Lọc các voxel thực sự có vận tốc (True Positives của dynamic objects)
            dynamic_mask = (occ_gt >= 2) & (occ_gt <= 10) & (occ_pred == occ_gt)
            if np.any(dynamic_mask):
                v_pred = flow_pred[:, dynamic_mask] # [3, N]
                v_gt = flow_gt[:, dynamic_mask]     # [3, N]
                # L2 norm của vector sai lệch vận tốc
                vel_err = np.linalg.norm(v_pred - v_gt, axis=0) # [N]
                self.velocity_errors.extend(vel_err.tolist())

    def compute(self):
        """Tính toán tổng hợp các chỉ số theo chuẩn CVPR Occ3D Benchmark"""
        free_idx = getattr(Config, 'FREE_LABEL', 17)

        # 1. Geometric Voxel IoU (Phân loại Nhị phân: Vật thể vs Không gian trống Free=17)
        occ_indices = [c for c in range(self.num_classes) if c != free_idx]
        tp_geo = np.sum(self.confusion_matrix[np.ix_(occ_indices, occ_indices)])
        fn_geo = np.sum(self.confusion_matrix[occ_indices, free_idx])
        fp_geo = np.sum(self.confusion_matrix[free_idx, occ_indices])
        voxel_iou = (tp_geo / (tp_geo + fp_geo + fn_geo + 1e-6)) * 100.0

        # 2. Semantic mIoU (Đánh giá trên 17 classes ngữ nghĩa, không tính Free=17)
        ious = []
        class_ious = {}
        for c in occ_indices:
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            union = tp + fp + fn
            iou_c = (tp / (union + 1e-6)) * 100.0 if union > 0 else 0.0
            ious.append(iou_c)
            class_ious[Config.CLASS_NAMES[c]] = iou_c

        miou = float(np.nanmean(ious))

        # 3. mAVE (mean Absolute Velocity Error)
        if len(self.velocity_errors) > 0:
            mave = float(np.mean(self.velocity_errors))
        else:
            mave = 0.5 # Giá trị mặc định

        # 4. OccScore (Equation 18 trong paper: OccScore = mIoU * 0.9 + max(1 - mAVE, 0) * 0.1 * 100)
        occ_score = (miou * 0.9) + (max(1.0 - mave, 0.0) * 10.0)

        # 5. Phân tích lỗi False Positive và False Negative
        fp_rate = (fp_geo / (tp_geo + fp_geo + 1e-6)) * 100.0
        fn_rate = (fn_geo / (tp_geo + fn_geo + 1e-6)) * 100.0

        return {
            'voxel_iou': voxel_iou,
            'miou': miou,
            'mave': mave,
            'occ_score': occ_score,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'class_ious': class_ious
        }
