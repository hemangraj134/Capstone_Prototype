import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

MODEL_NAME = "microsoft/codebert-base"

class CodeBERTClassifier(nn.Module):
    def __init__(self, num_labels=20): # Increased labels for Top CWEs
        super().__init__()
        # 1. Tokenizer: Standard CodeBERT BPE
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 2. Backbone: Encoder-only Transformer
        # We load it with safetensors for security/speed
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
        hidden_size = self.backbone.config.hidden_size
        
        # 3. Classification Head
        # Dropout prevents the model from "memorizing" specific CVE strings (overfitting)
        self.dropout = nn.Dropout(0.3) 
        self.classifier = nn.Linear(hidden_size, num_labels)

        # 4. LoRA Configuration (The Laptop Optimizer)
        # We target query, key, and value to capture structural code patterns
        config = LoraConfig(
            r=16, # Rank: Higher = more "intelligence" but uses more VRAM
            lora_alpha=32,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION 
        )
        
        # 5. PEFT Wrap: Only 1.2% of weights remain trainable
        self.backbone = get_peft_model(self.backbone, config)

    def forward(self, input_ids, attention_mask):
        # Extract features from the [CLS] token (represents the whole function)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits