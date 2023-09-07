import math
import random

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule

from .config import get_config
from .utils import greedy_decode

config = get_config()


class LayerNormalization(LightningModule):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1))  # bias is a learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, seq_len, d_model)
        # Keep the dimension for broadcasting
        mean = x.mean(-1, keepdim=True)  # (batch_size, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(-1, keepdim=True)  # (batch_size, seq_len, 1)
        # eps is to prevent division by zero
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(LightningModule):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(LightningModule):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings as per the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(LightningModule):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create matrix of shape (seq_len, d_model) to store positional encodings
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len,)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        # Create a vector of shape (d_model,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        # Apply sine to even indices of the positional encoding
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * (10000^(2i/d_model)))
        # Apply cosine to odd indices of the positional encoding
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)
        # Register the buffer so it will be moved to a device
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.pe[:, : x.size(1), :].requires_grad_(
            False
        )  # (batch_size, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(LightningModule):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: LightningModule) -> torch.Tensor:
        # x.shape == sublayer(x).shape
        # Apply residual connection to any sublayer with the same size
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(LightningModule):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.h = h  # Number of heads
        self.d_model = d_model  # Embedding Dimension

        # Make sure d_model is a multiple of h
        assert d_model % h == 0, "d_model must be a multiple of h"

        self.d_k = d_model // h  # Dimension of vectors after head split
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ):
        d_k = query.size(-1)
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -infinity after softmax) to the attention scores where the mask is 0
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # Softmax along the last dimension
        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        # Return attention scores and the result of the attention operation
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(
            q
        )  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(
            k
        )  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.w_v(
            v
        )  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(
            1, 2
        )

        # Apply attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # Combine all heads together again
        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)

        # Multiply by Wo
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.w_o(x)


class EncoderBlock(LightningModule):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        return self.residual_connections[1](x, self.feed_forward_block)


class Encoder(LightningModule):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(LightningModule):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(LightningModule):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(
        self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask, tgt_mask
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(LightningModule):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_position: PositionalEncoding,
        tgt_position: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # (batch_size, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_position(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, vocab_size)
        return self.projection_layer(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"], eps=1e-9)
        return optimizer

    def loss_fn(self, proj_output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # (batch_size * seq_len, vocab_size), (batch_size * seq_len)
        return nn.CrossEntropyLoss(
            ignore_index=self.trainer.datamodule.tokenizer_tgt.token_to_id("[PAD]"),
            label_smoothing=0.1,
        )(proj_output, label)

    def training_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"]  # (b, seq_len)
        decoder_input = batch["decoder_input"]  # (B, seq_len)
        encoder_mask = batch["encoder_mask"]  # (B, 1, 1, seq_len)
        decoder_mask = batch["decoder_mask"]  # (B, 1, seq_len, seq_len)

        # Run the tensors through the encoder, decoder, and the projection layers
        encoder_output = self.encode(
            encoder_input, encoder_mask
        )  # (B, seq_len, d_model)
        decoder_output = self.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )  # (B, seq_len, d_model)
        proj_output = self.project(decoder_output)  # (B, seq_len, vocab_tgt_len)

        # Compare the output with the label
        label = batch["label"]  # (B, seq_len)
        loss = self.loss_fn(
            proj_output.view(
                -1, self.trainer.datamodule.tokenizer_tgt.get_vocab_size()
            ),
            label.view(-1),
        )
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def on_validation_epoch_start(self):
        self.source_texts = []
        self.expected = []
        self.predicted = []

    def on_validation_epoch_end(self):
        # Print 4 random samples

        for _ in range(4):
            idx = random.randint(0, len(self.source_texts) - 1)
            source_text = self.source_texts[idx]
            target_text = self.expected[idx]
            model_out_text = self.predicted[idx]
            # Print the source, target and model output
            print("-" * 80)
            print(f"{f'SOURCE: '}{source_text}")
            print(f"{f'TARGET: '}{target_text}")
            print(f"{f'PREDICTED: '}{model_out_text}")

        # Char Error Rate
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(self.predicted, self.expected)
        self.log("Validation CER", cer, prog_bar=True, logger=True)

        # Word Error Rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(self.predicted, self.expected)
        self.log("Validation WER", wer, prog_bar=True, logger=True)

        # BLEU Score
        metric = torchmetrics.text.BLEUScore()
        bleu = metric(self.predicted, self.expected)
        self.log("Validation BLEU", bleu, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"]  # (batch_size, seq_len)
        encoder_mask = batch["encoder_mask"]  # (batch_size, 1, 1, seq_len)
        # check that the batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

        model_out = greedy_decode(
            self,
            encoder_input,
            encoder_mask,
            self.trainer.datamodule.tokenizer_src,
            self.trainer.datamodule.tokenizer_tgt,
            config["seq_len"],
        )

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = self.trainer.datamodule.tokenizer_tgt.decode(
            model_out.detach().cpu().numpy()
        )

        self.source_texts.append(source_text)
        self.expected.append(target_text)
        self.predicted.append(model_out_text)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the transformer blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(
            EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        )

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(
            DecoderBlock(
                decoder_self_attention_block,
                decoder_cross_attention_block,
                feed_forward_block,
                dropout,
            )
        )

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
