import os
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from config.config import config

class SkeletonDataset(Dataset):
    def __init__(self, file_list, data_dir, num_frames=100, mode='train'):
        self.file_list = file_list
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        try:
            # Nâng cấp việc đọc file .mat
            try:
                mat_data = sio.loadmat(file_path, verify_compressed_data_integrity=False)
            except:
                mat_data = sio.loadmat(file_path)

            # Kiểm tra cấu trúc skeleton_data
            if 'skeleton' not in mat_data:
                # Thử các khóa khác nếu không có 'skeleton'
                skeleton_keys = [key for key in mat_data.keys() if 'skeleton' in key.lower()]
                if skeleton_keys:
                    skeleton_data = mat_data[skeleton_keys[0]]
                else:
                    print(f"Không tìm thấy skeleton trong file {file_path}")
                    return self._create_empty_data()
            else:
                skeleton_data = mat_data['skeleton']

            # Xử lý label
            try:
                label = int(self.file_list[idx][1:4]) - 1
            except:
                print(f"Không thể trích xuất label từ {self.file_list[idx]}")
                label = 0

            # Xử lý dữ liệu skeleton
            processed_data = self._preprocess_skeleton(skeleton_data)

            # Trích xuất các loại đặc trưng
            joint_data = processed_data
            bone_data = self._get_bone_features(processed_data)
            velocity_data = self._get_velocity_features(processed_data)

            return joint_data, bone_data, velocity_data, label

        except Exception as e:
            print(f"Lỗi xử lý file {file_path}: {e}")
            return self._create_empty_data()

    def _create_empty_data(self):
        """Tạo dữ liệu trống khi không thể đọc file"""
        empty_data = torch.zeros((config['NUM_FRAMES'], config['NUM_JOINTS'], 2), dtype=torch.float32)
        return empty_data, empty_data, empty_data, 0

    def _preprocess_skeleton(self, skeleton_data):
        # Preprocessing logic
        if skeleton_data.shape[0] == 1:
            data = skeleton_data[0]
        else:
            data = skeleton_data[0]

        num_frames = data.shape[0]
        if num_frames >= self.num_frames:
            if self.mode == 'train':
                start = np.random.randint(0, num_frames - self.num_frames + 1)
                data = data[start:start+self.num_frames]
            else:
                start = (num_frames - self.num_frames) // 2
                data = data[start:start+self.num_frames]
        else:
            padding = np.tile(data[-1:], (self.num_frames - num_frames, 1, 1))
            data = np.concatenate([data, padding], axis=0)

        data = data[:, :, :2]

        center_joint = data[:, 1:2, :]
        data = data - center_joint

        scale_reference = np.sqrt(np.sum(np.square(
            data[:, 1, :] - data[:, 2, :]), axis=1))
        scale_reference = np.mean(scale_reference)
        if scale_reference != 0:
            data = data / scale_reference

        return torch.tensor(data, dtype=torch.float32)

    def _get_bone_features(self, joint_data):
        # Bone features logic
        connections = [
            (1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6),
            (8, 2), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12),
            (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
            (19, 18), (20, 1), (21, 20), (22, 21), (23, 22), (24, 1)
        ]

        bone_data = torch.zeros_like(joint_data)

        for j, (target, source) in enumerate(connections):
            if j < config['NUM_JOINTS'] - 1:
                bone_data[:, j, :] = joint_data[:, target, :] - joint_data[:, source, :]

        return bone_data

    def _get_velocity_features(self, joint_data):
        velocity = torch.zeros_like(joint_data)
        velocity[1:] = joint_data[1:] - joint_data[:-1]
        return velocity