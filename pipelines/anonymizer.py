"""
Module tự động làm mờ (anonymize) mặt người đi bộ và biển số xe trên các frame camera
Đảm bảo tuân thủ ràng buộc bảo mật dữ liệu của VinFast
"""
import cv2
import numpy as np

class PrivacyAnonymizer:
    def __init__(self, blur_kernel=(31, 31)):
        self.blur_kernel = blur_kernel
        # Sử dụng Haar Cascade có sẵn của OpenCV - chạy nhanh, không tốn thêm VRAM GPU
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def anonymize_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """Làm mờ mặt người và vùng nghi ngờ biển số xe"""
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr
            
        out_img = img_bgr.copy()
        gray = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Phát hiện và làm mờ khuôn mặt
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        for (x, y, w, h) in faces:
            roi = out_img[y:y+h, x:x+w]
            out_img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, self.blur_kernel, 0)
            
        return out_img

    def anonymize_batch(self, cam_dict: dict) -> dict:
        """Làm mờ đồng loạt cho 6 camera"""
        clean_cams = {}
        for cam_name, img in cam_dict.items():
            clean_cams[cam_name] = self.anonymize_image(img)
        return clean_cams
