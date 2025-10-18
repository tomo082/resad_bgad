# datasets/visa.py

import os
import cv2
import glob
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import imgaug.augmenters as iaa
import albumentations as A
#from .perlin import rand_perlin_2d_np
#from .nsa import patch_ex
from .utils_bgad import excluding_images


VISA_CLASS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum'
]


class VisaDataset(Dataset):
    def __init__(self, c, is_train=True, excluded_images=None):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, VISA_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        
        # load dataset
        self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()
        if excluded_images is not None:
            self.x, self.y, self.mask, self.img_types = excluding_images(self.x, self.y, self.mask, self.img_types, excluded_images)

        # set transforms
        self.transform_x = T.Compose([
            T.Resize(c.img_size, Image.ANTIALIAS),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])
        
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        img_path, y, mask, img_type = self.x[idx], self.y[idx], self.mask[idx], self.img_types[idx]
        
        x = Image.open(img_path).convert('RGB')
        x = self.normalize(self.transform_x(x))
        
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return x, y, mask, os.path.basename(img_path), img_type

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask, types = [], [], [], []
        
        img_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'Images')
        mask_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'Masks', 'Anomaly')

        # 학습 시에는 정상 이미지만 사용
        if self.is_train:
            normal_dir = os.path.join(img_dir, 'Normal')
            img_fpath_list = [os.path.join(normal_dir, f) for f in sorted(os.listdir(normal_dir)) if f.endswith('.JPG')]
            x.extend(img_fpath_list)
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))
            types.extend(['normal'] * len(img_fpath_list))
        # 테스트 시에는 정상과 비정상 이미지를 모두 사용
        else:
            normal_dir = os.path.join(img_dir, 'Normal')
            anomaly_dir = os.path.join(img_dir, 'Anomaly')
            
            # 정상 이미지 로드
            normal_fpath_list = [os.path.join(normal_dir, f) for f in sorted(os.listdir(normal_dir)) if f.endswith('.JPG')]
            x.extend(normal_fpath_list)
            y.extend([0] * len(normal_fpath_list))
            mask.extend([None] * len(normal_fpath_list))
            types.extend(['normal'] * len(normal_fpath_list))
            
            # 비정상 이미지 로드
            anomaly_fpath_list = [os.path.join(anomaly_dir, f) for f in sorted(os.listdir(anomaly_dir)) if f.endswith('.JPG')]
            x.extend(anomaly_fpath_list)
            y.extend([1] * len(anomaly_fpath_list))
            
            # 비정상 이미지에 대한 마스크 로드
            mask_fpath_list = [os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + '.png') for f in anomaly_fpath_list]
            mask.extend(mask_fpath_list)
            types.extend(['anomaly'] * len(anomaly_fpath_list))

        assert len(x) == len(y) == len(mask) == len(types)
        return x, y, mask, types


class VisaFSDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies
        self.reuse_times = 5
        
        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        self.a_imgs = self.a_imgs * self.reuse_times
        self.a_labels = self.a_labels * self.reuse_times
        self.a_masks = self.a_masks * self.reuse_times
        
        self.transform_x = T.Compose([
            T.Resize(c.img_size, Image.ANTIALIAS),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])

        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        if idx >= len(self.n_imgs):
            idx_ = idx - len(self.n_imgs)
            img, label, mask = self.a_imgs[idx_], self.a_labels[idx_], self.a_masks[idx_]
        else:
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]

        x = Image.open(img).convert('RGB')
        x = self.normalize(self.transform_x(x))
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return x, label, mask

    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []
        a_img_paths, a_labels, a_mask_paths = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'Images')
        mask_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'Masks', 'Anomaly')

        # 정상 이미지 로드 (학습용)
        normal_dir = os.path.join(img_dir, 'Normal')
        img_fpath_list = [os.path.join(normal_dir, f) for f in sorted(os.listdir(normal_dir)) if f.endswith('.JPG')]
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))

        # 비정상 이미지 로드 (Few-shot용 샘플)
        anomaly_dir = os.path.join(img_dir, 'Anomaly')
        anomaly_fpath_list = [os.path.join(anomaly_dir, f) for f in sorted(os.listdir(anomaly_dir)) if f.endswith('.JPG')]
        random.shuffle(anomaly_fpath_list)
        
        num_anomalies_to_take = min(self.anomaly_nums, len(anomaly_fpath_list))
        a_img_paths.extend(anomaly_fpath_list[:num_anomalies_to_take])
        a_labels.extend([1] * num_anomalies_to_take)
        
        mask_fpath_list = [os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + '.png') for f in anomaly_fpath_list[:num_anomalies_to_take]]
        a_mask_paths.extend(mask_fpath_list)

        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class VisaFSCopyPasteDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies
        self.repeat_num = 10
        self.reuse_times = 5

        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        
        self.a_imgs_real = self.a_imgs * self.reuse_times
        self.a_labels_real = self.a_labels * self.reuse_times
        self.a_masks_real = self.a_masks * self.reuse_times

        self.a_imgs_generated = self.a_imgs * (self.repeat_num - self.reuse_times)
        self.a_labels_generated = self.a_labels * (self.repeat_num - self.reuse_times)
        self.a_masks_generated = self.a_masks * (self.repeat_num - self.reuse_times)

        self.transform_img = T.Compose([
            T.Resize(c.img_size, Image.ANTIALIAS),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])
            
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])
        
        self.augmentors = [A.RandomRotate90(), A.Flip(), A.Transpose(),
                           A.OneOf([A.GaussNoise(), A.GaussNoise()], p=0.2),
                           A.OneOf([A.MotionBlur(p=.2), A.MedianBlur(blur_limit=3, p=0.1), A.Blur(blur_limit=3, p=0.1)], p=0.2),
                           A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                           A.OneOf([A.OpticalDistortion(p=0.3), A.GridDistortion(p=.1), A.PiecewiseAffine(p=0.3)], p=0.2),
                           A.OneOf([A.CLAHE(clip_limit=2), A.Sharpen(), A.Emboss(), A.RandomBrightnessContrast()], p=0.3),
                           A.HueSaturationValue(p=0.3)]
                           
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])
    
    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs_real) + len(self.a_imgs_generated)

    def __getitem__(self, idx):
        if idx < len(self.n_imgs):
            # 정상 샘플
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]
            img = Image.open(img).convert('RGB')
        elif idx < len(self.n_imgs) + len(self.a_imgs_real):
            # 실제 비정상 샘플
            idx_ = idx - len(self.n_imgs)
            img, label, mask = self.a_imgs_real[idx_], self.a_labels_real[idx_], self.a_masks_real[idx_]
            img = Image.open(img).convert('RGB')
        else:
            # Copy-Paste로 생성된 비정상 샘플
            idx_ = idx - len(self.n_imgs) - len(self.a_imgs_real)
            img_path, label, mask_path = self.a_imgs_generated[idx_], self.a_labels_generated[idx_], self.a_masks_generated[idx_]
            img, mask = self.copy_paste(img_path, mask_path)
            img, mask = Image.fromarray(img), Image.fromarray(mask)

        img = self.normalize(self.transform_img(img))
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            if isinstance(mask, str):
                mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return img, label, mask

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentors)), 3, replace=False)
        aug = A.Compose([self.augmentors[aug_ind[0]], self.augmentors[aug_ind[1]], self.augmentors[aug_ind[2]]])
        return aug

# datasets/visa.py の VisaFSCopyPasteDataset.copy_paste を修正

    def copy_paste(self, img_path, mask_path):
        n_idx = np.random.randint(len(self.n_imgs))
        aug = self.randAugmenter()

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        n_image = cv2.imread(self.n_imgs[n_idx])
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)
        
        # ===== 画像リサイズ処理を追加 =====
        image = cv2.resize(image, dsize=(self.cropsize[1], self.cropsize[0]))
        n_image = cv2.resize(n_image, dsize=(self.cropsize[1], self.cropsize[0]))
        # ==================================

        mask = np.asarray(Image.open(mask_path).convert('L'))
        mask = cv2.resize(mask, dsize=(self.cropsize[1], self.cropsize[0])) # マスクもリサイズ
        
        augmented = aug(image=image, mask=mask)
        aug_image, aug_mask = augmented['image'], augmented['mask']
        
        aug_mask = np.where(aug_mask > 0, 255, 0).astype(np.uint8)

        n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
        return n_image, aug_mask
        

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []
        a_img_paths, a_labels, a_mask_paths = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'Images')
        mask_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'Masks', 'Anomaly')

        # 정상 이미지 로드 (학습용)
        normal_dir = os.path.join(img_dir, 'Normal')
        img_fpath_list = [os.path.join(normal_dir, f) for f in sorted(os.listdir(normal_dir)) if f.endswith('.JPG')]
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))

        # 비정상 이미지 로드 (Copy-Paste 소스용)
        anomaly_dir = os.path.join(img_dir, 'Anomaly')
        anomaly_fpath_list = [os.path.join(anomaly_dir, f) for f in sorted(os.listdir(anomaly_dir)) if f.endswith('.JPG')]
        random.shuffle(anomaly_fpath_list)

        num_anomalies_to_take = min(self.anomaly_nums, len(anomaly_fpath_list))
        a_img_paths.extend(anomaly_fpath_list[:num_anomalies_to_take])
        a_labels.extend([1] * num_anomalies_to_take)
        
        mask_fpath_list = [os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + '.png') for f in anomaly_fpath_list[:num_anomalies_to_take]]
        a_mask_paths.extend(mask_fpath_list)

        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class VisaPseudoDataset(Dataset):
    def __init__(self, c, is_train=True):
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies
        self.repeat_num = 10
        
        self.n_imgs, _, _ , _, _, _ = self.load_dataset_folder()
        self.a_imgs = self.n_imgs * self.repeat_num
        
        self.transform_img_np = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crop_size)])
        
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(c.norm_mean, c.norm_std)])
        self.anomaly_source_paths = sorted(glob.glob(c.anomaly_source_path + "/*/*.jpg"))
        
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                           iaa.Solarize(0.5, threshold=(32,128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]], self.augmenters[aug_ind[2]]])
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.cropsize[1], self.cropsize[0]))
        anomaly_img_augmented = aug(image=anomaly_source_img)
        
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.cropsize[0], self.cropsize[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        beta = torch.rand(1).numpy()[0] * 0.8
        
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        
        has_anomaly = 1 if np.sum(msk) > 0 else 0
        return augmented_image, msk, has_anomaly

    def transform_image(self, image_path, anomaly_source_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform_img_np(image)
        image = np.asarray(image)
        
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = augmented_image.astype(np.uint8)
        
        return self.normalize(augmented_image), torch.from_numpy(np.transpose(anomaly_mask, (2, 0, 1))), has_anomaly

    def __getitem__(self, idx):
        if idx < len(self.n_imgs):
            img_path, label, mask = self.n_imgs[idx], 0, torch.zeros([1, self.cropsize[0], self.cropsize[1]])
            image = self.normalize(T.ToTensor()(Image.open(img_path).convert('RGB')))
            return image, label, mask
        else:
            n_idx = np.random.randint(len(self.n_imgs))
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            image, mask, label = self.transform_image(self.n_imgs[n_idx], self.anomaly_source_paths[anomaly_source_idx])
            return image, label, mask

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []
        a_img_paths, a_labels, a_mask_paths = [], [], []
        
        img_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'Images', 'Normal')
        img_fpath_list = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.endswith('.JPG')]
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))
        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class VisaAnomalyDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES
        self.copy_paste_dataset = VisaFSCopyPasteDataset(c, is_train=is_train)
        self.pseudo_dataset = VisaPseudoDataset(c, is_train=is_train)

    def __len__(self):
        return len(self.copy_paste_dataset) + len(self.pseudo_dataset) - len(self.copy_paste_dataset.n_imgs)

    def __getitem__(self, idx):
        if idx < len(self.pseudo_dataset):
             return self.pseudo_dataset.__getitem__(idx)
        else:
             return self.copy_paste_dataset.__getitem__(idx - len(self.pseudo_dataset) + len(self.copy_paste_dataset.n_imgs))
