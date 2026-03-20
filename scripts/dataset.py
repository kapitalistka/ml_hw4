import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, Optional
import timm


import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class CaloriesDatasetV2(Dataset):
   
    def __init__(
        self,
        dish_df: pd.DataFrame,
        ingredients_df: pd.DataFrame,
        images_dir: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True
    ):
        self.dish_df = dish_df.reset_index(drop=True)
        self.ingredients_df = ingredients_df
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.is_train = is_train
        
        self.ingr_id_to_name = dict(
            zip(
                ingredients_df['id'].astype(str),
                ingredients_df['ingr'].astype(str)
            )
        )
    
    def _get_ingredient_names(self, ingredients_str: str) :
        
        parts = ingredients_str.split(';')
        names = []
        
        for part in parts:
            ingr_id = part.strip()
            if ingr_id in self.ingr_id_to_name:
                names.append(self.ingr_id_to_name[ingr_id])
    
        
        return ', '.join(names)
    
    def __len__(self):
        return len(self.dish_df)
    
    def __getitem__(self, idx: int):
        row = self.dish_df.iloc[idx]
        
        dish_id = str(row['dish_id'])
        image_path = os.path.join(self.images_dir, dish_id, 'rgb.png')
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            #image = self.transform(image)
            image = self.transform(image=np.array(image))["image"]

        
        ingredients_str = row.get('ingredients', '')
        ingredient_text = self._get_ingredient_names(ingredients_str)
        
        # Токенизация
        encoding = self.tokenizer(
            ingredient_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        mass = row.get('total_mass', 1.0)
        calories = row.get('total_calories', 0.0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image': image,
            'mass': torch.tensor(mass, dtype=torch.float32),
            'calories': torch.tensor(calories, dtype=torch.float32),
            'dish_id': dish_id,
            'ingredient_text': ingredient_text
        }
    


def get_transforms_v2(is_train: bool, image_size: int, image_model_name: str):

    cfg = timm.get_pretrained_cfg(image_model_name)
    
    if is_train:
        return A.Compose([
            A.SmallestMaxSize(max_size=image_size + 32, p=1.0),
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.15, 
                rotate_limit=20, 
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
                A.ISONoise(p=0.5),
            ], p=0.3),
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1, 
                p=0.5
         ),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.ToTensorV2()
    ])
    else:
        return A.Compose(
            [

                A.SmallestMaxSize(max_size=image_size),
                A.CenterCrop(height=image_size, width=image_size),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2()

            ]
        )
    

def create_dataloaders_v2(config, tokenizer):
    ingredients_df = pd.read_csv(config['ingredients_path'])
    dish_df = pd.read_csv(config['dish_path'])
    
    train_val_df = dish_df[dish_df['split'] == 'train'].copy()
    test_df = dish_df[dish_df['split'] == 'test'].copy()
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config['val_ratio'],
        random_state=config['random_seed']
    )
    
    print(f'Размер выборок:')
    print(f'  Train: {len(train_df)}')
    print(f'  Val: {len(val_df)}')
    print(f'  Test: {len(test_df)}')
    
    train_dataset = CaloriesDatasetV2(
        dish_df=train_df,
        ingredients_df=ingredients_df,
        images_dir=config['images_dir'],
        tokenizer=tokenizer,
        max_length=128,
        transform=get_transforms_v2(is_train=True, image_size=config['image_size'], image_model_name = config['image_model_name']),
        is_train=True
    )
    
    val_dataset = CaloriesDatasetV2(
        dish_df=val_df,
        ingredients_df=ingredients_df,
        images_dir=config['images_dir'],
        tokenizer=tokenizer,
        max_length=128,
        transform=get_transforms_v2(is_train=False, image_size=config['image_size'], image_model_name = config['image_model_name']),
        is_train=False
    )
    
    test_dataset = CaloriesDatasetV2(
        dish_df=test_df,
        ingredients_df=ingredients_df,
        images_dir=config['images_dir'],
        tokenizer=tokenizer,
        max_length=128,
        transform=get_transforms_v2(is_train=False, image_size=config['image_size'], image_model_name = config['image_model_name']),
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader