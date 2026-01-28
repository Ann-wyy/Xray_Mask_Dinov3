import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Union
from PIL import Image
from monai.transforms import Resize as MonaiResize
from torchvision import transforms
import random

class RandomFlipRotate2D:
    """
    同步增强 2D X-ray + mask
    - 随机水平翻转
    - 随机旋转 ±15°
    """
    def __init__(self, flip_prob: float = 0.5, max_angle: float = 15):
        self.flip_prob = flip_prob
        self.max_angle = max_angle

    def __call__(self, image: np.ndarray, masks):
        """
        image: np.ndarray, shape (1,H,W) or (H,W)
        masks: dict of np.ndarray (each (H,W)) or single np.ndarray (H,W)
        """
        # 确保 image 是 (H,W)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]

        # 检查masks是字典还是单个数组
        is_dict = isinstance(masks, dict)

        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = np.fliplr(image)
            if is_dict:
                for k in masks:
                    masks[k] = np.fliplr(masks[k])
            else:
                masks = np.fliplr(masks)

        # 随机旋转
        angle = random.uniform(-self.max_angle, self.max_angle)
        # 使用 PIL 旋转（双线性插值 image, 最近邻 mask）
        from PIL import Image
        pil_img = Image.fromarray((image*255).astype(np.uint8))
        pil_img = pil_img.rotate(angle, resample=Image.BILINEAR)
        image = np.array(pil_img, dtype=np.float32)/255.0

        if is_dict:
            for k in masks:
                pil_mask = Image.fromarray(masks[k].astype(np.uint8))
                pil_mask = pil_mask.rotate(angle, resample=Image.NEAREST)
                masks[k] = np.array(pil_mask, dtype=np.int64)
        else:
            pil_mask = Image.fromarray(masks.astype(np.uint8))
            pil_mask = pil_mask.rotate(angle, resample=Image.NEAREST)
            masks = np.array(pil_mask, dtype=np.int64)

        # 恢复 channel 维度 (1,H,W)
        image = np.expand_dims(image, 0)
        if is_dict:
            for k in masks:
                masks[k] = np.expand_dims(masks[k], 0)
        else:
            masks = np.expand_dims(masks, 0)

        return image, masks

class XrayBoneDataset(Dataset):
    """
    X-ray骨平片数据集（保留分割mask + 多任务标签）

    Args:
        image_dir: 图像目录
        mask_dir: mask目录（.npz或.png文件）
        label_file: CSV标签文件路径，或直接传入字典 {patient_id: {task_name: label}}
        target_shape: 输出影像和mask大小 (H, W)
        transform: 数据增强
        mode: 'train', 'val', 'test'
        use_preprocessed: 是否跳过归一化和resize
        single_mask: 是否使用单一骨骼mask（True）还是多器官mask字典（False）
        mask_key: 当使用.npz且single_mask=True时，指定要加载的mask key（默认'bone'）
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        label_file: Union[str, Dict[str, Dict[str, int]]],
        target_shape: tuple = (512, 512),
        transform=None,
        mode: str = 'train',
        use_preprocessed: bool = False,
        single_mask: bool = True,
        mask_key: str = 'bone'
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_shape = target_shape
        self.transform = transform
        self.mode = mode
        self.use_preprocessed = use_preprocessed
        self.single_mask = single_mask
        self.mask_key = mask_key

        # 加载标签：支持CSV文件或字典
        if isinstance(label_file, dict):
            self.label_dict = label_file
        else:
            # 从CSV加载
            self.labels_df = pd.read_csv(label_file)
            self.label_dict = self._load_labels_from_csv()

        # MONAI 2D resize transform
        self.resize_transform = MonaiResize(spatial_size=target_shape, mode="bilinear")

        self.patient_ids = list(self.label_dict.keys())
        print(f"[{mode}] 加载了 {len(self.patient_ids)} 个样本")

    def _load_labels_from_csv(self) -> Dict[str, Dict[str, int]]:
        """从CSV加载标签并转换为字典格式"""
        label_dict = {}
        for _, row in self.labels_df.iterrows():
            patient_id = str(row['patient_id'])
            labels = {}
            for col in self.labels_df.columns:
                if col != 'patient_id':
                    labels[col] = int(row[col])
            label_dict[patient_id] = labels
        return label_dict

    def __len__(self):
        return len(self.patient_ids)

    def _load_image(self, patient_id: str):
        path = os.path.join(self.image_dir, f"{patient_id}.png")  # 或 .jpg
        image = np.array(Image.open(path).convert('L'), dtype=np.float32)

        if not self.use_preprocessed:
            # 简单归一化到 [0,1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)

        # resize
        image = np.expand_dims(image, 0)  # (1, H, W)
        image = self.resize_transform(image)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        return image.squeeze(0)

    def _load_mask(self, patient_id: str):
        """
        加载mask：支持单一骨骼mask或多器官mask字典
        - single_mask=True: 返回单个np.ndarray (H,W)
        - single_mask=False: 返回字典 {organ: np.ndarray (H,W)}
        """
        # 尝试加载.npz文件
        npz_path = os.path.join(self.mask_dir, f"{patient_id}.npz")
        png_path = os.path.join(self.mask_dir, f"{patient_id}.png")

        if os.path.exists(npz_path):
            # 从.npz加载
            data = np.load(npz_path, allow_pickle=True)
            if 'masks' in data:
                mask_dict = data['masks'].item()  # {'bone': mask, ...}
            elif 'mask' in data:
                mask_dict = {'bone': data['mask']}
            else:
                # 尝试直接使用第一个key
                mask_dict = {k: data[k] for k in data.keys()}
        elif os.path.exists(png_path):
            # 从.png加载单一mask
            mask = np.array(Image.open(png_path).convert('L'), dtype=np.float32)
            mask = (mask > 127).astype(np.int64)  # 二值化
            mask_dict = {self.mask_key: mask}
        else:
            raise FileNotFoundError(f"Mask文件不存在: {npz_path} 或 {png_path}")

        if self.single_mask:
            # 返回单一mask
            if self.mask_key in mask_dict:
                mask = mask_dict[self.mask_key]
            else:
                # 如果指定的key不存在，使用第一个mask
                mask = list(mask_dict.values())[0]

            # Resize
            mask = np.expand_dims(mask, 0)  # (1, H, W)
            mask_resized = self.resize_transform(mask).squeeze(0)
            if isinstance(mask_resized, torch.Tensor):
                mask_resized = mask_resized.numpy()
            mask_resized = (mask_resized > 0.5).astype(np.int64)
            return mask_resized  # (H, W)
        else:
            # 返回多器官mask字典
            resized_masks = {}
            for k, mask in mask_dict.items():
                mask = np.expand_dims(mask, 0)  # (1, H, W)
                mask_resized = self.resize_transform(mask).squeeze(0)
                if isinstance(mask_resized, torch.Tensor):
                    mask_resized = mask_resized.numpy()
                mask_resized = (mask_resized > 0.5).astype(np.int64)
                resized_masks[k] = mask_resized
            return resized_masks

    def _get_labels(self, patient_id: str):
        """
        使用初始化传入的字典生成标签
        """
        if patient_id not in self.label_dict:
            raise KeyError(f"Patient ID {patient_id} 不在 label_dict 中")
        return self.label_dict[patient_id]

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]

        # image
        image = self._load_image(patient_id)
        # mask
        masks = self._load_mask(patient_id)
        # labels
        labels = self._get_labels(patient_id)

        # 数据增强
        if self.transform is not None and self.mode == 'train':
            image, masks = self.transform(image, masks)

        # 确保image是2D (H,W)，因为RandomFlipRotate2D会添加channel维度
        if isinstance(image, np.ndarray) and image.ndim == 3:
            image = image.squeeze(0)

        # 转tensor
        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float()  # (1,H,W)

        if self.single_mask:
            # 确保mask是2D (H,W)
            if isinstance(masks, np.ndarray) and masks.ndim == 3:
                masks = masks.squeeze(0)
            masks_tensor = {self.mask_key: torch.from_numpy(np.ascontiguousarray(masks)).unsqueeze(0).long()}
        else:
            # 多器官mask字典，确保每个mask是2D
            masks_tensor = {}
            for k, v in masks.items():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    v = v.squeeze(0)
                masks_tensor[k] = torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).long()

        labels_tensor = {k: torch.tensor(v).long() for k, v in labels.items()}

        return {
            'image': image,
            'masks': masks_tensor,
            'labels': labels_tensor,
            'patient_id': patient_id
        }


# ========================= 测试示例 =========================
if __name__ == "__main__":
    # 模拟标签字典
    label_dict = {
        'patient001': {'femur_fracture': 1, 'pelvis_fracture': 0},
        'patient002': {'femur_fracture': 0, 'pelvis_fracture': 0},
    }
    train_transform = RandomFlipRotate2D(flip_prob=0.5, max_angle=15)

    dataset = XrayBoneDataset(
        image_dir="./images",
        mask_dir="./masks",
        label_dict=label_dict,
        target_shape=(512, 512),
        transform=train_transform,
        mode='train'
    )

    print(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    print(f"患者ID: {sample['patient_id']}")
    print(f"影像形状: {sample['image'].shape}")
    print(f"标签: {sample['labels']}")
    print(f"Mask keys: {list(sample['masks'].keys())}")

    """| patient_id | img_path        | mask_path      | fracture | dislocation |
| ---------- | --------------- | -------------- | -------- | ----------- |
| 0001       | images/0001.png | masks/0001.png | 1        | 0           |
| 0002       | images/0002.png | masks/0002.png | 0        | 1           |
"""