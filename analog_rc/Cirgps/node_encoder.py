import torch
import torch.nn as nn


DEFAULT_CONFIG = {
    "dev": {
        "discrete": [(0, 2), (1, 2), (2, 2), (5, 2), (6, 2), (15, 2)],
        "continuous": [3, 4, 7, 8, 9, 10, 11, 12, 13, 14],
    },
    "pin": {
        "discrete": [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)],
        "continuous": [],
    },
    "net": {
        "discrete": [(0, 2), (1, 2), (2, 2), (3, 2), (4, 550), (5, 313), (6, 385), (7, 547)],
        "continuous": [8, 9],
    },
}

NODE_TYPE_NAME_TO_ID = {"dev": 0, "pin": 1, "net": 2}


def expand_indices(spec):
    if isinstance(spec, tuple) and len(spec) == 2:
        start, end = spec
        return list(range(start, end))
    if isinstance(spec, list):
        return spec
    return []


class NodeFeatureEncoder(nn.Module):
    def __init__(self, out_dim, feat_config=None, default_vocab_size=100):
        super().__init__()
        self.out_dim = out_dim
        self.feat_config = feat_config or DEFAULT_CONFIG
        self.default_vocab_size = default_vocab_size

        self.encoders = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.parsed_configs = {}

        for node_type, type_config in self.feat_config.items():
            discrete_indices, discrete_vocab_sizes = self._parse_discrete_config(
                type_config.get("discrete", [])
            )
            continuous_indices = expand_indices(type_config.get("continuous", []))

            self.parsed_configs[node_type] = {
                "discrete_indices": discrete_indices,
                "discrete_vocab_sizes": discrete_vocab_sizes,
                "continuous_indices": continuous_indices,
            }

            type_encoders, encoder_out_dim = self._create_type_encoder(
                discrete_indices,
                discrete_vocab_sizes,
                continuous_indices,
            )
            self.encoders[node_type] = type_encoders
            if encoder_out_dim > 0:
                self.projections[node_type] = nn.Linear(encoder_out_dim, out_dim)

    def _parse_discrete_config(self, spec):
        indices = []
        vocab_sizes = []

        if isinstance(spec, tuple) and len(spec) == 2:
            start, end = spec
            indices = list(range(start, end))
            vocab_sizes = [self.default_vocab_size] * len(indices)
        elif isinstance(spec, list):
            for item in spec:
                if isinstance(item, tuple) and len(item) == 2:
                    idx, vocab_size = item
                    indices.append(idx)
                    vocab_sizes.append(vocab_size)
                elif isinstance(item, int):
                    indices.append(item)
                    vocab_sizes.append(self.default_vocab_size)

        return indices, vocab_sizes

    def _create_type_encoder(self, discrete_indices, discrete_vocab_sizes, continuous_indices):
        encoders = nn.ModuleDict()
        total_out_dim = 0

        for i, vocab_size in enumerate(discrete_vocab_sizes):
            key = f"discrete_{i}"
            embed_dim = max(self.out_dim // 4, 16)
            encoders[key] = nn.Embedding(vocab_size, embed_dim)
            total_out_dim += embed_dim

        if continuous_indices:
            feat_dim = len(continuous_indices)
            cont_out_dim = max(self.out_dim // 2, 32)
            encoders["continuous"] = nn.Sequential(
                nn.Linear(feat_dim, cont_out_dim),
                nn.ReLU(),
            )
            total_out_dim += cont_out_dim

        return encoders, total_out_dim

    def _prepare_discrete_feature(self, feat, vocab_size):
        if torch.is_floating_point(feat):
            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            feat_min = feat.min()
            feat_max = feat.max()
            if feat_min >= -1e-6 and feat_max <= 1.0 + 1e-6 and vocab_size > 1:
                feat = feat * float(vocab_size - 1)
        return feat.round().long().clamp(0, vocab_size - 1)

    def forward(self, node_attr, node_type):
        device = node_attr.device
        output = torch.zeros(node_attr.size(0), self.out_dim, device=device)

        for type_name, type_id in NODE_TYPE_NAME_TO_ID.items():
            if type_name not in self.feat_config:
                continue

            mask = node_type == type_id
            if not mask.any():
                continue

            type_attr = node_attr[mask]
            parsed = self.parsed_configs[type_name]
            encoders = self.encoders[type_name]
            encoded_parts = []

            for i, (idx, vocab_size) in enumerate(
                zip(parsed["discrete_indices"], parsed["discrete_vocab_sizes"])
            ):
                key = f"discrete_{i}"
                feat = self._prepare_discrete_feature(type_attr[:, idx], vocab_size)
                encoded_parts.append(encoders[key](feat))

            continuous_indices = parsed["continuous_indices"]
            if continuous_indices and "continuous" in encoders:
                feat = torch.nan_to_num(
                    type_attr[:, continuous_indices],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                encoded_parts.append(encoders["continuous"](feat))

            if encoded_parts:
                type_encoded = torch.cat(encoded_parts, dim=1)
                output[mask] = self.projections[type_name](type_encoded)

        return output
