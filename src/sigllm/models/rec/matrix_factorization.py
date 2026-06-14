import torch    
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_index = 0
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=self.padding_index)
        # Bias terms (CTR-style MF). Improve fit under class imbalance and give
        # cold items/users a learnable offset. They do not affect user/item
        # encoder outputs, so the downstream Q-Former pipeline is unaffected.
        self.user_bias = nn.Embedding(config.user_num, 1, padding_idx=self.padding_index)
        self.item_bias = nn.Embedding(config.item_num, 1, padding_idx=self.padding_index)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        # Default nn.Embedding init is N(0, 1). With a dot product over
        # `embedding_size` dims that produces logits with std ~= sqrt(dim),
        # which saturates BCEWithLogitsLoss and kills gradients (AUC stuck at
        # chance). Small-std init keeps initial logits near 0 so training can
        # actually start.
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.user_embedding.weight[self.padding_index].zero_()
            self.item_embedding.weight[self.padding_index].zero_()

    def user_encoder(self,user_ids):
        return self.user_embedding(user_ids)

    def item_encoder(self,item_ids):
        return self.item_embedding(item_ids)

    def compute(self):
        return None, None

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embedding(user_ids)
        item_embeddings = self.item_embedding(item_ids)
        matching = torch.mul(user_embeddings, item_embeddings).sum(dim=-1)
        matching = matching + self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1) + self.global_bias
        return matching