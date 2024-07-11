import torch
from transformers import AutoModel, AutoTokenizer  # Assuming Hugging Face Transformers

# Load pre-trained MiniLM model
minilm_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') 

# Build the E5small model
class E5Small(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.transformer = minilm_model  # Use the pre-trained encoder
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        self.contrastive_head = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_ids):
        embeddings = self.embedding_layer(input_ids)
        transformer_output = self.transformer(inputs_embeds=embeddings) 
        # Usually, take the [CLS] token output or a mean-pooling for representation
        pooled_output = transformer_output[0][:, 0, :]  # Example
        embeddings = self.contrastive_head(pooled_output)
        return embeddings

print(minilm_model.config)
vocab_size = minilm_model.config.vocab_size
hidden_size = minilm_model.config.hidden_size
embedding_model = E5Small(vocab_size=vocab_size, embedding_dim=hidden_size)
print(embedding_model)