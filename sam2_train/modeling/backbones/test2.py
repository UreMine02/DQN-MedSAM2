import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class PointEmbeddingExtractor(nn.Module):
    def __init__(self, embed_dim: int = 256, input_image_size: Tuple[int, int] = (1024, 1024)):
        super(PointEmbeddingExtractor, self).__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Initialize the point embeddings for 2 labels (as in your code)
        self.point_embeddings = nn.Embedding(2, embed_dim)

    def forward(self, mask: torch.Tensor, image_embedding: torch.Tensor, batch_size: int, num_points: int) -> torch.Tensor:
        """
        Extract the coordinates of points where mask = 1, and then extract the corresponding embeddings
        from image_embedding.
        
        mask: tensor of shape [1, 1, 256, 256] or [1, 1, 1024, 1024] - Binary mask (0 or 1)
        image_embedding: tensor of shape [1, 256, 64, 64] - Image embedding
        batch_size: Batch size of the input
        num_points: Number of points to sample
        
        Returns:
        point_embeddings: tensor of shape [num_points, 256] - Point embeddings at the coordinates
        """
        # Get coordinates where mask is 1
        mask = mask.squeeze(0)  # Remove batch dimension, shape: [256, 256] or [1024, 1024]
        coords = torch.nonzero(mask == 1, as_tuple=False)  # Find coordinates with mask value 1
        
        if coords.size(0) < num_points:
            sample = np.random.choice(coords.size(0), num_points, replace=True)
        else:
            sample = np.random.choice(coords.size(0), num_points, replace=False)
        
        # Sample coordinates
        x = coords[sample, 0].unsqueeze(1)  # (num_points, 1)
        y = coords[sample, 1].unsqueeze(1)  # (num_points, 1)
        points = torch.cat([x, y], dim=1).unsqueeze(1).float()  # (num_points, 1, 2)
        points_torch = points.to(image_embedding.device)

        # Embed the points (using PositionEmbeddingRandom)
        point_embeddings = self.pe_layer.forward_with_coords(points_torch, self.input_image_size)

        # Assuming labels are random for this example (you can modify it based on your application)
        labels = torch.randint(0, 2, (num_points, 1), device=image_embedding.device)  # Random labels 0 or 1
        point_embeddings[labels == 0] += self.point_embeddings.weight[0]  # Add embedding for label 0
        point_embeddings[labels == 1] += self.point_embeddings.weight[1]  # Add embedding for label 1
        
        return point_embeddings


# Example usage:
# Example mask and image_embedding (random values)
mask_1024 = torch.ones(1, 1, 256, 256)  # All ones in the mask
image_embedding = torch.randn(1, 256, 64, 64)  # Example image embedding (random values)

batch_size = 1
num_points = 999  # Number of points to sample

# Initialize the extractor
extractor = PointEmbeddingExtractor(embed_dim=256)

# Extract point embeddings
point_embeddings = extractor(mask_1024, image_embedding, batch_size, num_points)

# Print the result
print(f"Point Embedding Shape: {point_embeddings.shape}")
