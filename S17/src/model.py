import math

import torch
import torch.nn as nn

"""
Super Transformer Model Class
This is the class that can be used for both the encoder and decoder
given appropriate parameters.
"""


class MultiheadSelfAttentionBlock(nn.Module):
    """
    A single multihead self-attention block.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
    ) -> None:
        """
        Initializes a single multihead self-attention block.
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            attn_dropout: Dropout probability for attention
        Returns:
            None
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single multihead self-attention block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attn_mask: Attention mask of shape (batch_size, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Layer norm
        x = self.layer_norm(x)

        # Multihead self-attention
        attn_output, _ = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
            attn_mask=attn_mask,
        )

        return attn_output


class MultiLayerPerceptronBlock(nn.Module):
    """
    A single multi-layer perceptron block.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        mlp_dropout: float,
        mlp_activation: nn.Module,
    ) -> None:
        """
        Initializes a single multi-layer perceptron block.

        Args:
            embed_dim: Embedding dimension
            mlp_dim: Hidden dimension of the MLP
            mlp_dropout: Dropout probability for MLP
        Returns:
            None
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            mlp_activation,
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        mlp_dim: int,
        mlp_dropout: float,
        mlp_activation: nn.Module,
    ) -> None:
        """
        Initializes a single transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            attn_dropout: Dropout probability for attention
            mlp_dim: Hidden dimension of the MLP
            mlp_dropout: Dropout probability for MLP
        Returns:
            None
        """
        super().__init__()

        self.mha_block = MultiheadSelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )

        self.mlp_block = MultiLayerPerceptronBlock(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            mlp_dropout=mlp_dropout,
            mlp_activation=mlp_activation,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention block with residual connection
        x = x + self.mha_block(x, attn_mask)

        # Multi-layer perceptron block with residual connection
        x = x + self.mlp_block(x)

        return x


# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, requires_grad: bool = False):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]  # x.size(1) = seq_len


class Transformer(nn.Module):
    """
    A transformer model.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        mlp_dim: int,
        mlp_dropout: float,
        mlp_activation: nn.Module,
        num_layers: int,
        embed_dict_size: int,
        max_seq_len: int,
        pad_idx: int,
        add_cls_token: bool = False,
        pe_requires_grad: bool = False,
        need_embedding: bool = True,
    ) -> None:
        """
        Initializes a transformer model.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            attn_dropout: Dropout probability for attention
            mlp_dim: Hidden dimension of the MLP
            mlp_dropout: Dropout probability for MLP
            mlp_activation: Activation function for MLP
            num_layers: Number of transformer blocks
            embded_dict_size: Size of the embedding dictionary
            pad_idx: Index of the padding token
        Returns:
            None
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.mlp_activation = mlp_activation
        self.num_layers = num_layers
        self.embed_dict_size = embed_dict_size
        self.pad_idx = pad_idx
        self.add_cls_token = add_cls_token
        self.max_seq_len = max_seq_len
        self.need_embedding = need_embedding

        if need_embedding:
            # Token embedding layer
            self.token_embed_layer = nn.Embedding(
                num_embeddings=embed_dict_size,
                embedding_dim=embed_dim,
                padding_idx=pad_idx,
            )

        # Positional embedding layer
        self.pos_embed_layer = PositionalEmbedding(
            d_model=embed_dim, max_seq_len=max_seq_len, requires_grad=pe_requires_grad
        )

        # Class Token embedding
        if add_cls_token:
            self.cls_embed_layer = nn.Embedding(
                num_embeddings=1,
                embedding_dim=embed_dim,
            )

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    mlp_dim=mlp_dim,
                    mlp_dropout=mlp_dropout,
                    mlp_activation=mlp_activation,
                )
                for _ in range(num_layers)
            ]
        )

        # Classifier head
        self.classifier_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if self.need_embedding:
            x = self.token_embed_layer(x)
        if self.add_cls_token:
            # Add CLS token
            cls_token = self.cls_embed_layer(
                torch.zeros(x.size(0), 1).long().to(x.device)
            )
            x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embed_layer(x)
        for block in self.transformer_blocks:
            x = block(x, attn_mask)

        x = self.classifier_layer_norm(x)
        return x
