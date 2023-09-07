import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, random_split

from .utils import causal_mask, get_or_build_tokenizer


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = (
            self.seq_len - len(enc_input_tokens) - 2
        )  # -2 for sos and eos
        # We will only add <s> to input, and </s> to the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative, if it is the input is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise Exception("Input is too long")

        encoder_input = torch.cat(
            (
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ),
            dim=0,
        )

        # Decoder input: Add only <s> token
        decoder_input = torch.cat(
            (
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ),
            dim=0,
        )

        # Label: Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        # Double check the tensors to make sure they are all seq_len long
        assert (
            len(encoder_input) == self.seq_len
            and len(decoder_input) == self.seq_len
            and len(label) == self.seq_len
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(decoder_input.size(0)).int(),  # (1, seq_len, seq_len),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


class BilingualDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        num_workers=2,
    ):
        super().__init__()
        self.config = config
        self.num_workers = num_workers

    def prepare_data(self):
        load_dataset(
            "opus_books",
            f'{self.config["lang_src"]}-{self.config["lang_tgt"]}',
            split="train",
        )

    def setup(self, stage=None):
        self.ds_raw = load_dataset(
            "opus_books",
            f'{self.config["lang_src"]}-{self.config["lang_tgt"]}',
            split="train",
        )

        # Build tokenizers
        self.tokenizer_src = get_or_build_tokenizer(
            self.config, self.ds_raw, self.config["lang_src"]
        )
        self.tokenizer_tgt = get_or_build_tokenizer(
            self.config, self.ds_raw, self.config["lang_tgt"]
        )

        # Keep 90% of the data for training, 10% for validation
        train_ds_size = int(0.9 * len(self.ds_raw))
        val_ds_size = len(self.ds_raw) - train_ds_size
        self.train_ds_raw, self.val_ds_raw = random_split(
            self.ds_raw, [train_ds_size, val_ds_size]
        )

        self.train_ds = BilingualDataset(
            self.train_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"],
        )
        self.val_ds = BilingualDataset(
            self.val_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"],
        )

        # Find the max length of the sentences in source and target language
        max_len_src = 0
        max_len_tgt = 0

        for item in self.ds_raw:
            max_len_src = max(
                len(
                    self.tokenizer_src.encode(
                        item["translation"][self.config["lang_src"]]
                    ).ids
                ),
                max_len_src,
            )
            max_len_tgt = max(
                len(
                    self.tokenizer_tgt.encode(
                        item["translation"][self.config["lang_tgt"]]
                    ).ids
                ),
                max_len_tgt,
            )

        print("Max length of source sentences:", max_len_src)
        print("Max length of target sentences:", max_len_tgt)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )
