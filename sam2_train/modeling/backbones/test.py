import torch
import torch.nn as nn

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
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
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
    def __init__(self):
        super(PointEmbeddingExtractor, self).__init__()
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

    def forward(self, mask, image_embedding):
        """
        Extract the coordinates of points where mask = 1, and then extract the corresponding embeddings
        from image_embedding.
        
        mask: tensor of shape [1, 1, 256, 256] or [1, 1, 1024, 1024] - Binary mask (0 or 1)
        image_embedding: tensor of shape [1, 256, 64, 64] - Image embedding
        
        Returns:
        point_embeddings: tensor of shape [num_points, 256] - Point embeddings at the coordinates
        """
        sample = np.random.choice(np.arange(l), num_points, replace=True)
        x = torch.where(gt_mask == 1)[2][sample].unsqueeze(1)  # (num_points, 1)
        y = torch.where(gt_mask == 1)[3][sample].unsqueeze(1)  # (num_points, 1)
        points = torch.cat([x, y], dim=1).unsqueeze(1).float() # (num_points, 1, 2)
        points_torch = points.to(self.device)
        points_torch = points_torch.transpose(0,1).repeat(batch_size, 1, 1)

    def _embed_points(
    self,
    points: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Embeds point prompts."""
    points = points + 0.5  # Shift to center of pixel
    padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
    padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
    points = torch.cat([points, padding_point], dim=1)
    labels = torch.cat([labels, padding_label], dim=1)
    point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
    point_embedding[labels == 0] += self.point_embeddings[0].weight
    point_embedding[labels == 1] += self.point_embeddings[1].weight
    return point_embedding

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) 
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels)

        return point_embeddings

