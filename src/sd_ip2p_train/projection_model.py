import torch

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        if embeds.ndim == 2:
            bs = 1
            seq_len = embeds.shape[0]
        elif embeds.ndim == 3:
            bs = embeds.shape[0]
            seq_len = embeds.shape[1]
        else:
            raise ValueError("Invalid input shape for image_embeds")

        clip_extra_context_tokens = self.proj(embeds).reshape(
            bs, seq_len * self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens