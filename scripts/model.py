import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoTokenizer


class CaloriesMultimodalModel(nn.Module):

    
    def __init__(
        self,
        text_model_name: str = 'distilbert-base-uncased',
        image_model_name: str = 'resnet50',
        emb_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        
        # Текстовая ветка
        self.text_model = AutoModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_model.config.hidden_size
        
        # Ветка для картинок
        self.image_model = timm.create_model(
            image_model_name,
            pretrained=True,
            num_classes=0
        )
        image_features = self.image_model.num_features
        
        # Заморозка 
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        for param in self.image_model.parameters():
            param.requires_grad = False
        
        # к одной размерности
        self.text_proj = nn.Linear(text_hidden_size, emb_dim)
        self.image_proj = nn.Linear(image_features, emb_dim)
        
        # объединение
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
        mass: torch.Tensor
    ) -> torch.Tensor:
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]
        
        image_features = self.image_model(image)
        
        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        
        combined = torch.cat([text_emb, image_emb, mass.unsqueeze(1)], dim=1)
        
        output = self.fusion(combined)
        
        return output.squeeze(1)
