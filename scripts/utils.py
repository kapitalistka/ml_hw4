import os
import random
import numpy as np
from tqdm import tqdm
import torch



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_mae = 0
    num_samples = 0
    
    for batch in tqdm(loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        mass = batch['mass'].to(device)
        calories = batch['calories'].to(device)
        
        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask, images, mass)
        loss = criterion(predictions, calories)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(calories)
        total_mae += torch.abs(predictions - calories).sum().item()
        num_samples += len(calories)
    
    return {
        'loss': total_loss / num_samples,
        'mae': total_mae / num_samples
    }



def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    num_samples = 0
    
    for batch in tqdm(loader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        mass = batch['mass'].to(device)
        calories = batch['calories'].to(device)
        
        predictions = model(input_ids, attention_mask, images, mass)
        loss = criterion(predictions, calories)
        
        total_loss += loss.item() * len(calories)
        total_mae += torch.abs(predictions - calories).sum().item()
        num_samples += len(calories)
    
    return {
        'loss': total_loss / num_samples,
        'mae': total_mae / num_samples
    }


def train_model(config, train_loader, val_loader, test_loader, tokenizer, model  ):
    set_seed(config['random_seed'])
    device = config['device']
    print(f'Устройство: {device}')
    
    model.to(device)
    
    # Параметры модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Всего параметров: {total_params:,}')
    print(f'Обучаемых параметров: {trainable_params:,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.L1Loss()
    
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    best_mae = float('inf')
    patience_counter = 0
    
    print('\n' + '=' * 50)
    print('Начало обучения')
    print('=' * 50)
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f'\nЭпоха {epoch}/{config["num_epochs"]}')
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Сохранение истории
        history['train_loss'].append(train_metrics['loss'])
        history['train_mae'].append(train_metrics['mae'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        
        print(f'  Train MAE: {train_metrics["mae"]:.2f}')
        print(f'  Val MAE: {val_metrics["mae"]:.2f}')
        
        # Сохранение лучшей модели
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            patience_counter = 0
            os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), config['model_save_path'])
            print(f'  Модель сохранена! MAE: {best_mae:.2f}')
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f'\nРанняя остановка на эпохе {epoch}')
                break
    
    return {
        'model': model,
        'history': history,
        'best_mae': best_mae,
        'test_loader': test_loader
    }