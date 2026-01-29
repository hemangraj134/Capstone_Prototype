# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

MODEL_NAME = "microsoft/codebert-base"

class CodeBERTClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        # Tokenizer for preprocessing (if needed in dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load base model (CodeBERT = RoBERTa variant)
        # ✅ FIX: Added use_safetensors=True to bypass the torch.load vulnerability check
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
        hidden = self.backbone.config.hidden_size
        
        # Add classification head
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden, num_labels)

        # LoRA configuration for parameter-efficient fine-tuning
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "key", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION  # ✅ correct for encoder-only models
        )
        self.backbone = get_peft_model(self.backbone, config)

    def forward(self, input_ids, attention_mask):
        # No 'labels' are passed here (RobertaModel doesn’t accept them)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Take [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Classify
        logits = self.classifier(pooled_output)
        return logits