import torch
import torch.nn.functional as F
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

    def forward(self, mask: torch.Tensor, image_embedding: torch.Tensor) -> torch.Tensor:
        """
        Extract the coordinates of points where mask = 1, and then extract the corresponding embeddings
        from image_embedding.
        
        mask: tensor of shape [1, 1, 256, 256] or [1, 1, 1024, 1024] - Binary mask (0 or 1)
        image_embedding: tensor of shape [1, 256, 64, 64] - Image embedding
        
        Returns:
        point_embeddings: tensor of shape [num_points, 256] - Point embeddings at the coordinates
        """
        # Get coordinates where mask is 1
        mask = mask.squeeze(0)  # Remove batch dimension, shape: [256, 256] or [1024, 1024]
        mask = mask.squeeze(0)

        # coords = torch.nonzero(mask < 10, as_tuple=False)  # Find coordinates with mask value 1

        mask_flat = mask.view(-1)
        mask_softmax = F.softmax(mask_flat, dim=0)
        mask_softmax_2d = mask_softmax.view(mask.shape)  # Shape: [H, W]
        # Proba = 0.5
        coords = torch.nonzero(mask_softmax_2d > 0.002, as_tuple=False)  # [num_points, 2]

        # Calculate the number of points where mask == 1
        num_points = coords.size(0)
        # print('numpoint', num_points)
        # Sample coordinates if necessary (in case the number of points is large)
        if num_points > 0:
            sample = np.random.choice(num_points, num_points, replace=False)  # Sample without replacement
        
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
        else:
            return torch.empty(0, self.embed_dim)  # If no points, return empty tensor

# class AttnLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(AttnLayer, self).__init__()
#         self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
#         self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

#     def forward(self, backbone_features, low_res_multimasks):
#         """
#         Perform self-attention on A, and then cross-attention between A' (output of self-attention) and B.
        
#         A: tensor of shape (batch_size, seq_len, embed_dim) -> query for self-attention
#         B: tensor of shape (batch_size, seq_len, embed_dim) -> key_value for cross-attention
#         """
#         # Self-attention on backbone_features
#         backbone_features = backbone_features.flatten(2).permute(2, 0, 1)  # Reshape to (seq_len, batch_size, embed_dim) for MultiheadAttention
#         attn_output_A, attn_output_weights_A = self.self_attention(backbone_features, backbone_features, backbone_features)
        
#         # Convert back to original shape
#         backbone_features_prime = attn_output_A.permute(1, 2, 0).reshape(1, 256, 64, 64)  # Shape: [1, 256, 64, 64]
        
#         # Cross-attention between backbone_features' (from self-attention) and low_res_multimasks (as key-value)
#         low_res_multimasks = low_res_multimasks.flatten(2).permute(2, 0, 1)  # Reshape B to (seq_len, batch_size, embed_dim)
#         cross_attn_output, cross_attn_output_weights = self.cross_attention(backbone_features_prime.flatten(2).permute(2, 0, 1), low_res_multimasks, low_res_multimasks)
        
#         # Convert cross-attention output back to the original shape
#         cross_attn_output = cross_attn_output.permute(1, 2, 0).reshape(1, 256, 64, 64)  # Shape: [1, 256, 64, 64]
        
#         return cross_attn_output

# class AttnLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(AttnLayer, self).__init__()
#         # Only cross-attention, no self-attention
#         self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

#     def forward(self, backbone_features, low_res_multimasks):
#         """
#         Perform cross-attention between backbone_features (query) and low_res_multimasks (key-value).
        
#         backbone_features: tensor of shape (batch_size, seq_len, embed_dim) -> query for cross-attention
#         low_res_multimasks: tensor of shape (batch_size, seq_len, embed_dim) -> key_value for cross-attention
#         """
#         # Reshape the tensors to (seq_len, batch_size, embed_dim) as required by MultiheadAttention
#         backbone_features = backbone_features.flatten(2).permute(2, 0, 1)  # Shape: (seq_len, batch_size, embed_dim)
#         low_res_multimasks = low_res_multimasks.flatten(2).permute(2, 0, 1)  # Shape: (seq_len, batch_size, embed_dim)
        
#         # Apply cross-attention
#         cross_attn_output, cross_attn_output_weights = self.cross_attention(backbone_features, low_res_multimasks, low_res_multimasks)
        
#         # Convert cross-attention output back to the original shape
#         cross_attn_output = cross_attn_output.permute(1, 2, 0).reshape(1, 256, 64, 64)  # Shape: [1, 256, 64, 64]
        
#         return cross_attn_output


# class X(torch.nn.Module):
#     def __init__(self):
#         super(X, self).__init__()
#         self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1)  # Output size: [1, 128, 128, 128]
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Output size: [1, 256, 64, 64]

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))  # Apply ReLU activation after the convolution
#         x = torch.relu(self.conv2(x))
#         return x
        

class AttnLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttnLayer, self).__init__()
        self.cross_attn_image_to_point = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn_point_to_image = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, backbone_features, low_res_multimasks):
        B, C, H, W = backbone_features.shape
        image_seq = backbone_features.flatten(2).permute(2, 0, 1)  # [4096, 1, 256]

        # Ensure point_seq is 3D
        if low_res_multimasks.dim() == 2:
            point_seq = low_res_multimasks.unsqueeze(1)  # [N_points, 1, 256]
        else:
            point_seq = low_res_multimasks  # [N_points, 1, 256]
            
        device = backbone_features.device  

        image_seq = image_seq.to(device)
        point_seq = point_seq.to(device)


        image_attended, _ = self.cross_attn_image_to_point(image_seq, point_seq, point_seq)
        point_attended, _ = self.cross_attn_point_to_image(point_seq, image_seq, image_seq)

        updated_image_seq = image_seq + image_attended
        updated_backbone_features = updated_image_seq.permute(1, 2, 0).reshape(B, C, H, W)

        return updated_backbone_features



class BackboneUpdates(torch.nn.Module):
    def __init__(
        self,
        embed_dim=256, 
        num_heads=4
    ):
        super().__init__()
        self.x = PointEmbeddingExtractor(embed_dim)
        self.attn_layer = AttnLayer(embed_dim, num_heads)

    def forward(self, backbone_features, low_res_multimasks):
        # low_res_multimasks [1, 1, 256, 256] backbone_features [1,256,64,64] -> [1,256,64,64]
        low_res_features = self.x(low_res_multimasks, backbone_features ) # [1,256,64,64]
        backbone_features = self.attn_layer(backbone_features, low_res_features)
        return backbone_features

