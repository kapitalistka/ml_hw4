import torch

CONFIG = {
    'ingredients_path': 'data/ingredients.csv',
    'dish_path': 'data/dish.csv',
    'images_dir': 'data/images',
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 1e-4,
    'val_ratio': 0.2,
    'random_seed': 42,
    'image_size': 224,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_save_path': 'models/best_model_v2.pth',
    'early_stopping_patience': 7
}