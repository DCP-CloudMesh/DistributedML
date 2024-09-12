import torch
import torch.nn as nn
import torch.nn.functional as F

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial15/Vision_Transformer.html

# # patch embedding for transformer model 
# # split image into patches and put them in the embedding space
# class PatchEmbedding(nn.Module):
#     def __init__(self, img_size, patch_size, in_channels, embed_size):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.n_patches = (img_size // patch_size) ** 2
#         self.in_channels = in_channels
#         self.embed_size = embed_size

#         self.flatten = nn.Flatten(2)
#         self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_size)

#     def forward(self, x):
#         x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         x = x.contiguous().view(x.size(0), -1, self.patch_size * self.patch_size * self.in_channels)
#         x = self.projection(x)
#         return x


# patch embedding for transformer model
# split image into patches and put them in the embedding space
# using CNN to do feature extraction
# class PatchEmbedding(nn.Module):
#     def __init__(self, img_size, patch_size, in_channels, embed_size):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.n_patches = (img_size // patch_size) ** 2
#         self.embed_size = embed_size

#         self.proj = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x)  # shape: [batch_size, embed_size, H', W']
#         x = x.flatten(2)  # shape: [batch_size, embed_size, n_patches]
#         x = x.transpose(1, 2)  # shape: [batch_size, n_patches, embed_size]
#         return x


# patch embedding for transformer model
# split image into patches and put them in the embedding space
# using CNN to do feature extraction
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_size=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_size = embed_size

        self.proj = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # shape: [batch_size, embed_size, H', W']
        x = x.flatten(2)  # shape: [batch_size, embed_size, n_patches]
        x = x.transpose(1, 2)  # shape: [batch_size, n_patches, embed_size]
        return x


# actual multi-head attention and feed forward network
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            # nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(attn_output + x)
        x = self.norm2(self.feed_forward(x) + x)
        return x


# combining the patch layer and multi-head attention layer
# class VisionTransformer(nn.Module):
#     def __init__(self, img_size, patch_size, in_channels, embed_size, num_layers, num_heads, num_classes, dropout):
#         super().__init__()
#         self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_size)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
#         self.positional_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.n_patches, embed_size))
#         self.layers = nn.ModuleList([TransformerBlock(embed_size, num_heads, dropout) for _ in range(num_layers)])
#         self.to_cls_token = nn.Identity()
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(embed_size),
#             nn.Linear(embed_size, embed_size),
#             nn.Linear(embed_size, num_classes),
#             nn.LogSoftmax(dim=1)
#         )

#     def forward(self, x):
#         x = self.patch_embedding(x)
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_token, x), dim=1)
#         x += self.positional_embedding
#         for layer in self.layers:
#             x = layer(x)
#         x = self.to_cls_token(x[:, 0])
#         return self.mlp_head(x)

# combining the patch layer and multi-head attention layer
# update with changes to patch embeddings
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_size, num_layers, num_heads, num_classes, dropout):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.n_patches, embed_size))
        self.layers = nn.ModuleList([TransformerBlock(embed_size, num_heads, dropout) for _ in range(num_layers)])
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, embed_size),
            nn.Linear(embed_size, num_classes),
            nn.Softmax(dim=1)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        for layer in self.layers:
            x = layer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


if __name__ == '__main__':
    # CIFAR10 model
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,      # CIFAR-10 images are RGB, so 3 input channels
        embed_size=64,     # Smaller embedding size for a smaller dataset
        num_layers=6,       # Fewer layers might be sufficient for CIFAR-10
        num_heads=8,        # Adjusted number of heads
        num_classes=10,     # CIFAR-10 has 10 classes
        dropout=0.1
    )

    example_input = torch.randn(16, 3, 32, 32)  # (batch_size, channels, height, width)
    example_target = torch.randint(0, 10, (16,))

    # Forward pass through the model
    output = model(example_input)
    print(output.shape)
    print(output[0])

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loss = criterion(output, example_target)
    print(loss)
