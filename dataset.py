from glob import glob
from os import path
import os
import torch
from typing import Optional
import math
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import random
from PIL import ImageFilter
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

class ImagesDataset(Dataset):

    def __init__(
            self,
            image_dir,
            width: int = 100,
            height: int = 100,
            dtype: Optional[type] = None,
            is_train: bool = True
    ):
        self.image_filepaths = []
        for f in glob(path.join(image_dir, "*.jpg")):
            try:
                with Image.open(f) as img:
                    image = np.array(img)
                    if len(image.shape) == 2 or image.shape[2] == 3:
                        self.image_filepaths.append(path.abspath(f))
                    else:
                        print(f"Skipping file {f} as it is not a valid image file.")
            except Exception as e:
                print(f"Error processing file {f}: {e}")

        self.image_filepaths = sorted(self.image_filepaths)
        class_filepath = [path.abspath(f) for f in glob(path.join(image_dir, "*.csv"))][0]
        self.filenames_classnames, self.classnames_to_ids = ImagesDataset.load_classnames(class_filepath)
        if width < 100 or height < 100:
            raise ValueError('width and height must be greater than or equal 100')
        self.width = width
        self.height = height
        self.dtype = dtype
        self.is_train = is_train
        self.augment_prob = 0.6
        self.transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])

    def augment_image(self, image):
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.squeeze().astype('uint8'))

        # Random horizontal flip
        if random.random() < 0.7:
            image = ImageOps.mirror(image)

        # Random vertical flip
        if random.random() < 0.7:
            image = ImageOps.flip(image)
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-45, 45)
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
        
        # Brightness, Contrast adjustments
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        # Perspective transform
        # Perspective transform
        """if random.random() < 0.6:
            width, height = image.size
            startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
            endpoints = [(random.randint(0, width // 10), random.randint(0, height // 10)),
                        (random.randint(width * 9 // 10, width), random.randint(0, height // 10)),
                        (random.randint(width * 9 // 10, width), random.randint(height * 9 // 10, height)),
                        (random.randint(0, width // 10), random.randint(height * 9 // 10, height))]
            coeffs = find_coeffs(endpoints, startpoints)
            image = image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)"""
        
        # Random crop and resize
        """if random.random() < 0.7:
            width, height = image.size
            crop_size = random.uniform(0.7, 0.9)
            crop_width = max(1, int(width * crop_size))
            crop_height = max(1, int(height * crop_size))
            left = random.randint(0, max(0, width - crop_width))
            top = random.randint(0, max(0, height - crop_height))
            right = min(width, left + crop_width)
            bottom = min(height, top + crop_height)
            image = image.crop((left, top, right, bottom))
            image = image.resize((width, height), Image.BICUBIC)"""
            
        # Random affine transformation
        # Random affine transformation
        """if random.random() < 0.6:
            angle = random.uniform(-20, 20)
            translate = (random.uniform(-0.1, 0.1) * image.width, 
                        random.uniform(-0.1, 0.1) * image.height)
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-10, 10)
            image = image.transform(
                image.size, 
                Image.AFFINE, 
                (1, shear / 100, -translate[0], shear / 100, 1, -translate[1]), 
                resample=Image.BILINEAR
            )
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.BILINEAR)
            # Crop back to original size
            left = (image.width - 100) // 2
            top = (image.height - 100) // 2
            right = left + 100
            bottom = top + 100
            image = image.crop((left, top, right, bottom))"""
            
        # Random gaussian blur
        if random.random() < 0.3:
            radius = random.uniform(0, 1)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Random erasing
        if random.random() < 0.2:
            np_image = np.array(image)
            h, w = np_image.shape
            erase_area = random.uniform(0.02, 0.2) * (h * w)
            aspect_ratio = random.uniform(0.3, 1/0.3)
            erase_h = min(h-1, max(1, int(round((erase_area * aspect_ratio) ** 0.5))))
            erase_w = min(w-1, max(1, int(round((erase_area / aspect_ratio) ** 0.5))))
            x = random.randint(0, w - erase_w)
            y = random.randint(0, h - erase_h)
            erase_value = random.randint(0, 255)
            np_image[y:y+erase_h, x:x+erase_w] = erase_value
            image = Image.fromarray(np_image)
        
        # Black bars augmentation
        # Black bars augmentation
        if random.random() < 0.5:
            np_image = np.array(image)
            h, w = np_image.shape
            max_bar_width = max(1, w // 6)
            max_bar_height = max(1, h // 6)
            bar_width = random.randint(1, max_bar_width)
            bar_height = random.randint(1, max_bar_height)
            
            num_bars = random.randint(1, 3)  # Randomly choose between 1 and 3 bars
            
            if random.random() < 0.5:
                # Horizontal black bars
                gap = (h - num_bars * bar_height) // (num_bars + 1)  # Calculate gap between bars
                for i in range(num_bars):
                    y = gap + i * (bar_height + gap)  # Calculate y position for each bar
                    np_image[y:y+bar_height, :] = 0
            else:
                # Vertical black bars
                gap = (w - num_bars * bar_width) // (num_bars + 1)  # Calculate gap between bars
                for i in range(num_bars):
                    x = gap + i * (bar_width + gap)  # Calculate x position for each bar
                    np_image[:, x:x+bar_width] = 0
            
            image = Image.fromarray(np_image)
        
        return image

    
    def get_labels(self):
        return [self.classnames_to_ids[self.filenames_classnames[i][1]] for i in range(len(self))]
    
    def __getitem__(self, index):
        classname = self.filenames_classnames[index][1]
        classid = self.classnames_to_ids[classname]

        # If we only need the label (e.g., for calculating class weights)
        if not hasattr(self, 'return_image') or not self.return_image:
            return classid

        with Image.open(self.image_filepaths[index]).convert('L') as im:
            image = im
        
        # Apply augmentation with a certain probability only for training set
        if self.is_train and random.random() < self.augment_prob:
            image = self.augment_image(image)
        
        # Apply transform (resize and convert to tensor)
        image = self.transform(image)
        
        return image, classid, classname, self.image_filepaths[index]
    
    @staticmethod
    def load_classnames(class_filepath: str):
        filenames_classnames = np.genfromtxt(class_filepath, delimiter=';', skip_header=1, dtype=str)
        classnames = np.unique(filenames_classnames[:, 1])
        classnames.sort()
        classnames_to_ids = {}
        for index, classname in enumerate(classnames):
            classnames_to_ids[classname] = index
        return filenames_classnames, classnames_to_ids

    def __len__(self):
        return len(self.image_filepaths)
    
def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")
    
    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    
    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255
    
    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]


def prepare_image(image: np.ndarray, width: int, height: int, x: int, y: int, size: int):
    if image.ndim < 3 or image.shape[-3] != 1:
        raise ValueError("image must have shape (1, H, W)")
    if width < 32 or height < 32 or size < 32:
        raise ValueError("width/height/size must be >= 32")
    if x < 0 or (x + size) > width:
        raise ValueError(f"x={x} and size={size} do not fit into the resized image width={width}")
    if y < 0 or (y + size) > height:
        raise ValueError(f"y={y} and size={size} do not fit into the resized image height={height}")
    
    image = image.copy()

    if image.shape[1] > height:
        image = image[:, (image.shape[1] - height) // 2: (image.shape[1] - height) // 2 + height, :]
    else: 
        image = np.pad(image, ((0, 0), ((height - image.shape[1])//2, math.ceil((height - image.shape[1])/2)), (0, 0)), mode='edge')
    
    if image.shape[2] > width:
        image = image[:, :, (image.shape[2] - width) // 2: (image.shape[2] - width) // 2 + width]
    else:
        image = np.pad(image, ((0, 0), (0, 0), ((width - image.shape[2])//2, math.ceil((width - image.shape[2])/2))), mode='edge')

    subarea = image[:, y:y + size, x:x + size]
    return image, subarea

def cutmix(batch, alpha=1.0):
    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    
    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    
    return data, targets, shuffled_targets, lam

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)