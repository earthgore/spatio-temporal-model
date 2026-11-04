"""
Static-Graph Spatio-Temporal Model for physics simulation on irregular point clouds.

Architecture:
    Input:  coords (N, 3), features (B, T, N, F)
    Output: (B, N, C)

Supports two spatial processing modes:
    - "transformer": Edge-conditioned Graph Transformer (TransformerConv)
    - "message_passing": Physics-aware Message Passing Network

Graph topology is precomputed once (static coordinates) and reused every forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, radius_graph, knn_graph
from torch_geometric.utils import to_undirected, coalesce
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Literal, Union
from enum import Enum
import numpy as np

try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

kall
# ================================================================
# Configuration
# ================================================================

class GraphType(str, Enum):
    RADIUS = "radius"
    KNN = "knn"
    DELAUNAY = "delaunay"


class SpatialMode(str, Enum):
    TRANSFORMER = "transformer"
    MESSAGE_PASSING = "message_passing"


class TemporalMode(str, Enum):
    GRU = "gru"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CONV1D = "conv1d"
    MEAN = "mean"


class TemporalAggregation(str, Enum):
    LAST = "last"
    MEAN = "mean"
    ATTENTION = "attention"


class OutputBound(str, Enum):
    NONE = "none"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    CLAMP = "clamp"
    HARDTANH = "hardtanh"


ACTIVATION_FUNCTIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "mish": nn.Mish,
}


def get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in ACTIVATION_FUNCTIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(ACTIVATION_FUNCTIONS.keys())}"
        )
    return ACTIVATION_FUNCTIONS[key]()


@dataclass
class StaticGraphSTConfig:
    """Configuration for Static-Graph Spatio-Temporal Model."""

    # --- Data dimensions ---
    in_features: int = 6
    out_channels: int = 3
    max_timesteps: int = 12

    # --- Graph construction ---
    graph_type: GraphType = GraphType.RADIUS
    graph_radius: float = 1.0
    graph_k: int = 16
    graph_max_neighbors: int = 32
    prune_long_edges: bool = True
    prune_quantile: float = 0.95

    # --- Spatial processing ---
    spatial_mode: SpatialMode = SpatialMode.TRANSFORMER
    hidden_dim: int = 64
    spatial_layers: int = 6
    num_heads: int = 4
    spatial_dropout: float = 0.1
    spatial_aggr: str = "mean"

    # --- Edge features ---
    edge_dim: int = 4  # dx, dy, dz, dist (computed automatically)
    edge_hidden_dim: int = 32
    use_rbf: bool = True
    rbf_num_bases: int = 16
    rbf_cutoff: float = 5.0

    # --- Temporal processing ---
    temporal_mode: TemporalMode = TemporalMode.GRU
    temporal_layers: int = 2
    temporal_dropout: float = 0.1
    temporal_aggregation: TemporalAggregation = TemporalAggregation.LAST

    # --- Coordinate embedding ---
    use_coord_embedding: bool = True
    coord_embed_dim: int = 32

    # --- Output ---
    decoder_hidden_dim: int = 64
    decoder_layers: int = 2
    output_bound: OutputBound = OutputBound.NONE
    clamp_min: float = 0.0
    clamp_max: float = 1.0

    # --- Training ---
    nonlinearity: str = "silu"
    identity_skip: bool = True

    # --- Node features (boundary flags, etc.) ---
    num_node_static_features: int = 0


# ================================================================
# Graph Construction (Precomputed)
# ================================================================

class GraphBuilder:
    """
    Builds graph topology from static coordinates.
    Called ONCE before training. Results are registered as model buffers.
    """

    @staticmethod
    def build(
        coords: torch.Tensor,
        config: StaticGraphSTConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph and compute edge features.

        Args:
            coords: (N, 3) node coordinates
            config: model configuration

        Returns:
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
        """
        coords = coords.float()

        if config.graph_type == GraphType.RADIUS:
            edge_index = GraphBuilder._build_radius(coords, config)
        elif config.graph_type == GraphType.KNN:
            edge_index = GraphBuilder._build_knn(coords, config)
        elif config.graph_type == GraphType.DELAUNAY:
            edge_index = GraphBuilder._build_delaunay(coords, config)
        else:
            raise ValueError(f"Unknown graph type: {config.graph_type}")

        # Make undirected and remove duplicates
        edge_index = to_undirected(edge_index)
        edge_index = coalesce(edge_index)

        # Compute edge features
        edge_attr = GraphBuilder._compute_edge_features(coords, edge_index, config)

        # Prune excessively long edges (optional)
        if config.prune_long_edges:
            edge_index, edge_attr = GraphBuilder._prune_edges(
                edge_index, edge_attr, config.prune_quantile
            )

        return edge_index, edge_attr

    @staticmethod
    def _build_radius(coords: torch.Tensor, config: StaticGraphSTConfig) -> torch.Tensor:
        return radius_graph(
            coords,
            r=config.graph_radius,
            max_num_neighbors=config.graph_max_neighbors,
            loop=False,
        )

    @staticmethod
    def _build_knn(coords: torch.Tensor, config: StaticGraphSTConfig) -> torch.Tensor:
        return knn_graph(coords, k=config.graph_k, loop=False)

    @staticmethod
    def _build_delaunay(coords: torch.Tensor, config: StaticGraphSTConfig) -> torch.Tensor:
        if not HAS_SCIPY:
            raise ImportError("scipy is required for Delaunay triangulation")

        coords_np = coords.cpu().numpy()

        # Use 2D Delaunay if Z variation is negligible
        z_range = coords_np[:, 2].max() - coords_np[:, 2].min()
        if z_range < 1e-6:
            tri = Delaunay(coords_np[:, :2])
        else:
            tri = Delaunay(coords_np)

        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edges.add((simplex[i], simplex[j]))
                    edges.add((simplex[j], simplex[i]))

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        return edge_index

    @staticmethod
    def _compute_edge_features(
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        config: StaticGraphSTConfig,
    ) -> torch.Tensor:
        row, col = edge_index
        diff = coords[col] - coords[row]  # (E, 3) — direction from source to target
        dist = torch.norm(diff, p=2, dim=-1, keepdim=True)  # (E, 1)

        features = [diff, dist]  # base: dx, dy, dz, ||d||

        if config.use_rbf:
            rbf = GraphBuilder._rbf_encoding(dist.squeeze(-1), config.rbf_num_bases, config.rbf_cutoff)
            features.append(rbf)

        return torch.cat(features, dim=-1)

    @staticmethod
    def _rbf_encoding(distances: torch.Tensor, num_bases: int, cutoff: float) -> torch.Tensor:
        """Radial Basis Function encoding of distances."""
        centers = torch.linspace(0.0, cutoff, num_bases, device=distances.device)
        width = (cutoff / num_bases) * 0.5
        return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))

    @staticmethod
    def _prune_edges(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        quantile: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove edges longer than a given quantile."""
        # Distance is at index 3 (after dx, dy, dz)
        distances = edge_attr[:, 3]
        threshold = torch.quantile(distances, quantile)
        mask = distances <= threshold
        return edge_index[:, mask], edge_attr[mask]


# ================================================================
# Edge Encoder
# ================================================================

class EdgeEncoder(nn.Module):
    """Encodes raw edge features (dx, dy, dz, dist, rbf...) into hidden dimension."""

    def __init__(self, raw_edge_dim: int, hidden_dim: int, nonlinearity: str = "silu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_edge_dim, hidden_dim),
            get_activation(nonlinearity),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(edge_attr)


# ================================================================
# Coordinate Embedding
# ================================================================

class CoordEmbedding(nn.Module):
    """Embeds 3D coordinates into hidden space using Fourier features + MLP."""

    def __init__(self, out_dim: int, num_frequencies: int = 8, nonlinearity: str = "silu"):
        super().__init__()
        self.num_frequencies = num_frequencies
        input_dim = 3 + 3 * 2 * num_frequencies  # raw + sin/cos for each frequency
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            get_activation(nonlinearity),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 3) -> (N, out_dim)"""
        feats = [coords]
        for i in range(self.num_frequencies):
            freq = 2.0 ** i * torch.pi
            feats.append(torch.sin(freq * coords))
            feats.append(torch.cos(freq * coords))
        x = torch.cat(feats, dim=-1)
        return self.mlp(x)


# ================================================================
# Temporal Encoders
# ================================================================

class TemporalGRUEncoder(nn.Module):
    """GRU encoder for temporal sequences. Treats N as batch."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*N, T, F)
        returns: (B*N, T, H)
        """
        output, _ = self.gru(x)
        return output


class TemporalLSTMEncoder(nn.Module):
    """LSTM encoder for temporal sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output


class TemporalTransformerEncoder(nn.Module):
    """Transformer encoder for temporal sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        max_timesteps: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_timesteps, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*N, T, F)
        returns: (B*N, T, H)
        """
        T = x.shape[1]
        x = self.input_proj(x) + self.pos_embed[:, :T, :]
        return self.encoder(x)


class TemporalConv1DEncoder(nn.Module):
    """1D convolution encoder for temporal sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            out_ch = hidden_dim
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*N, T, F)
        returns: (B*N, T, H)
        """
        x = x.transpose(1, 2)  # (B*N, F, T)
        x = self.net(x)  # (B*N, H, T)
        return x.transpose(1, 2)  # (B*N, T, H)


class TemporalMeanEncoder(nn.Module):
    """Simple linear projection (no temporal modeling, just projects features)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ================================================================
# Temporal Aggregation (T -> 1)
# ================================================================

class TemporalLastAggregation(nn.Module):
    """Take the last timestep."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, :]


class TemporalMeanAggregation(nn.Module):
    """Average over time."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class TemporalAttentionAggregation(nn.Module):
    """Learnable query aggregation (cross-attention with single query)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*N, T, H)
        returns: (B*N, H)
        """
        q = self.query.expand(x.shape[0], -1, -1)  # (B*N, 1, H)
        out, _ = self.cross_attn(q, x, x)  # (B*N, 1, H)
        return self.norm(out.squeeze(1))  # (B*N, H)


# ================================================================
# Spatial Layers: Message Passing
# ================================================================

class MessagePassingLayer(nn.Module):
    """
    Physics-aware message passing layer with edge features.

    Message: MLP(concat(h_i, h_j, edge_attr))
    Update:  h_i = h_i + MLP(concat(h_i, aggregated_messages))
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        aggr: str = "mean",
        nonlinearity: str = "silu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.aggr = aggr

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            get_activation(nonlinearity),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(nonlinearity),
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            get_activation(nonlinearity),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (N_total, D) node features (could be B*N if batched)
        edge_index: (2, E)
        edge_attr: (E, edge_dim)
        returns: (N_total, D)
        """
        row, col = edge_index  # row -> col (source -> target)

        # Compute messages
        msg_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        messages = self.message_mlp(msg_input)  # (E, hidden_dim)

        # Aggregate messages per target node
        num_nodes = x.size(0)
        if self.aggr == "mean":
            aggr_messages = self._aggregate_mean(messages, col, num_nodes)
        elif self.aggr == "sum":
            aggr_messages = self._aggregate_sum(messages, col, num_nodes)
        elif self.aggr == "max":
            aggr_messages = self._aggregate_max(messages, col, num_nodes)
        else:
            aggr_messages = self._aggregate_mean(messages, col, num_nodes)

        # Update node features with residual
        update_input = torch.cat([x, aggr_messages], dim=-1)
        x = x + self.update_mlp(update_input)
        x = self.norm(x)

        return x

    def _aggregate_mean(self, messages: torch.Tensor, target: torch.Tensor, num_nodes: int) -> torch.Tensor:
        out = torch.zeros(num_nodes, messages.size(-1), device=messages.device, dtype=messages.dtype)
        count = torch.zeros(num_nodes, 1, device=messages.device, dtype=messages.dtype)
        out.scatter_add_(0, target.unsqueeze(-1).expand_as(messages), messages)
        count.scatter_add_(0, target.unsqueeze(-1), torch.ones_like(target.unsqueeze(-1), dtype=messages.dtype))
        return out / count.clamp(min=1)

    def _aggregate_sum(self, messages: torch.Tensor, target: torch.Tensor, num_nodes: int) -> torch.Tensor:
        out = torch.zeros(num_nodes, messages.size(-1), device=messages.device, dtype=messages.dtype)
        out.scatter_add_(0, target.unsqueeze(-1).expand_as(messages), messages)
        return out

    def _aggregate_max(self, messages: torch.Tensor, target: torch.Tensor, num_nodes: int) -> torch.Tensor:
        out = torch.full((num_nodes, messages.size(-1)), -float("inf"), device=messages.device, dtype=messages.dtype)
        out.scatter_reduce_(0, target.unsqueeze(-1).expand_as(messages), messages, reduce="amax")
        out = out.clamp(min=0)
        return out


# ================================================================
# Spatial Layers: Graph Transformer
# ================================================================

class GraphTransformerLayer(nn.Module):
    """
    Edge-conditioned Graph Transformer layer using PyG TransformerConv.

    Attention scores are modulated by edge features (distance, direction).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        nonlinearity: str = "silu",
    ):
        super().__init__()

        assert node_dim % num_heads == 0, f"node_dim ({node_dim}) must be divisible by num_heads ({num_heads})"

        self.conv = TransformerConv(
            in_channels=node_dim,
            out_channels=node_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=True,  # Learnable skip connection gate
        )

        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            get_activation(nonlinearity),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 4, node_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (N_total, D)
        edge_index: (2, E)
        edge_attr: (E, edge_dim)
        returns: (N_total, D)
        """
        # Pre-norm Graph Attention
        h = self.norm1(x)
        h = self.conv(h, edge_index, edge_attr)
        x = x + h

        # Pre-norm FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x


# ================================================================
# Spatial Processor (Stack of layers)
# ================================================================

class SpatialProcessor(nn.Module):
    """
    Stack of spatial layers (either Message Passing or Graph Transformer).
    """

    def __init__(self, config: StaticGraphSTConfig, edge_hidden_dim: int):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(config.spatial_layers):
            if config.spatial_mode == SpatialMode.TRANSFORMER:
                layer = GraphTransformerLayer(
                    node_dim=config.hidden_dim,
                    edge_dim=edge_hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.spatial_dropout,
                    nonlinearity=config.nonlinearity,
                )
            elif config.spatial_mode == SpatialMode.MESSAGE_PASSING:
                layer = MessagePassingLayer(
                    node_dim=config.hidden_dim,
                    edge_dim=edge_hidden_dim,
                    hidden_dim=config.hidden_dim,
                    aggr=config.spatial_aggr,
                    nonlinearity=config.nonlinearity,
                    dropout=config.spatial_dropout,
                )
            else:
                raise ValueError(f"Unknown spatial mode: {config.spatial_mode}")

            self.layers.append(layer)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


# ================================================================
# Output Decoder
# ================================================================

class OutputDecoder(nn.Module):
    """MLP decoder: hidden_dim -> out_channels."""

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 2,
        nonlinearity: str = "silu",
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                get_activation(nonlinearity),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ================================================================
# Main Model
# ================================================================

class StaticGraphSpatioTemporalModel(nn.Module):
    """
    Static-Graph Spatio-Temporal Model for physics simulation.

    Graph topology is precomputed from static coordinates and stored as buffers.
    Temporal processing happens per-node, spatial processing via graph layers.

    Args:
        coords: (N, 3) static node coordinates
        config: StaticGraphSTConfig
        node_static_features: (N, S) optional static per-node features
            (e.g., boundary flags, permeability, porosity)
    """

    def __init__(
        self,
        coords: torch.Tensor,
        config: StaticGraphSTConfig,
        node_static_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.config = config
        N = coords.shape[0]
        self.num_nodes = N

        # ============================================================
        # 1. PRECOMPUTE GRAPH (done once, stored as buffers)
        # ============================================================
        edge_index, edge_attr_raw = GraphBuilder.build(coords, config)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr_raw", edge_attr_raw)
        self.register_buffer("coords", coords.float())

        if node_static_features is not None:
            self.register_buffer("node_static_features", node_static_features.float())
        else:
            self.node_static_features = None

        # ============================================================
        # 2. EDGE ENCODER
        # ============================================================
        raw_edge_dim = edge_attr_raw.shape[1]
        self.edge_encoder = EdgeEncoder(raw_edge_dim, config.edge_hidden_dim, config.nonlinearity)

        # ============================================================
        # 3. COORDINATE EMBEDDING (optional)
        # ============================================================
        if config.use_coord_embedding:
            self.coord_embed = CoordEmbedding(config.coord_embed_dim, nonlinearity=config.nonlinearity)
        else:
            self.coord_embed = None

        # ============================================================
        # 4. TEMPORAL ENCODER
        # ============================================================
        temporal_input_dim = config.in_features
        self.temporal_encoder = self._build_temporal_encoder(temporal_input_dim, config)

        # ============================================================
        # 5. TEMPORAL AGGREGATION (T -> 1)
        # ============================================================
        self.temporal_aggregation = self._build_temporal_aggregation(config)

        # ============================================================
        # 6. NODE PROJECTION (temporal_output + coords + static -> hidden_dim)
        # ============================================================
        node_input_dim = config.hidden_dim
        if config.use_coord_embedding:
            node_input_dim += config.coord_embed_dim
        if self.node_static_features is not None:
            node_input_dim += config.num_node_static_features

        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, config.hidden_dim),
            get_activation(config.nonlinearity),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # ============================================================
        # 7. SPATIAL PROCESSOR
        # ============================================================
        self.spatial_processor = SpatialProcessor(config, config.edge_hidden_dim)

        # ============================================================
        # 8. OUTPUT DECODER
        # ============================================================
        self.decoder = OutputDecoder(
            hidden_dim=config.hidden_dim,
            out_channels=config.out_channels,
            num_layers=config.decoder_layers,
            nonlinearity=config.nonlinearity,
            dropout=config.spatial_dropout,
        )

        # ============================================================
        # 9. OUTPUT BOUNDING
        # ============================================================
        self.output_bound = config.output_bound
        self.clamp_min = config.clamp_min
        self.clamp_max = config.clamp_max

        # ============================================================
        # INIT WEIGHTS
        # ============================================================
        self._init_weights()

        # ============================================================
        # LOG GRAPH STATISTICS
        # ============================================================
        self._log_graph_stats()

    def _build_temporal_encoder(self, input_dim: int, config: StaticGraphSTConfig) -> nn.Module:
        if config.temporal_mode == TemporalMode.GRU:
            return TemporalGRUEncoder(input_dim, config.hidden_dim, config.temporal_layers, config.temporal_dropout)
        elif config.temporal_mode == TemporalMode.LSTM:
            return TemporalLSTMEncoder(input_dim, config.hidden_dim, config.temporal_layers, config.temporal_dropout)
        elif config.temporal_mode == TemporalMode.TRANSFORMER:
            return TemporalTransformerEncoder(
                input_dim, config.hidden_dim, config.temporal_layers,
                config.num_heads, config.max_timesteps, config.temporal_dropout,
            )
        elif config.temporal_mode == TemporalMode.CONV1D:
            return TemporalConv1DEncoder(input_dim, config.hidden_dim, config.temporal_layers, config.temporal_dropout)
        elif config.temporal_mode == TemporalMode.MEAN:
            return TemporalMeanEncoder(input_dim, config.hidden_dim)
        else:
            raise ValueError(f"Unknown temporal mode: {config.temporal_mode}")

    def _build_temporal_aggregation(self, config: StaticGraphSTConfig) -> nn.Module:
        if config.temporal_aggregation == TemporalAggregation.LAST:
            return TemporalLastAggregation()
        elif config.temporal_aggregation == TemporalAggregation.MEAN:
            return TemporalMeanAggregation()
        elif config.temporal_aggregation == TemporalAggregation.ATTENTION:
            return TemporalAttentionAggregation(config.hidden_dim, config.num_heads, config.temporal_dropout)
        else:
            raise ValueError(f"Unknown temporal aggregation: {config.temporal_aggregation}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _log_graph_stats(self):
        E = self.edge_index.shape[1]
        N = self.num_nodes

        # Degree statistics
        row = self.edge_index[0]
        degree = torch.zeros(N, dtype=torch.long, device=self.edge_index.device)
        degree.scatter_add_(0, row, torch.ones_like(row))

        # Edge length statistics
        dist = self.edge_attr_raw[:, 3] if self.edge_attr_raw.shape[1] > 3 else torch.norm(
            self.edge_attr_raw[:, :3], dim=-1
        )

        print(f"{'='*60}")
        print(f"Graph Statistics:")
        print(f"  Nodes: {N}")
        print(f"  Edges: {E} (avg degree: {E / N:.1f})")
        print(f"  Degree: min={degree.min()}, max={degree.max()}, "
              f"mean={degree.float().mean():.1f}, std={degree.float().std():.1f}")
        print(f"  Edge length: min={dist.min():.4f}, max={dist.max():.4f}, "
              f"mean={dist.mean():.4f}, std={dist.std():.4f}")
        print(f"  Spatial mode: {self.config.spatial_mode}")
        print(f"  Temporal mode: {self.config.temporal_mode}")
        print(f"  Temporal aggregation: {self.config.temporal_aggregation}")
        print(f"{'='*60}")

    def _apply_output_bound(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_bound == OutputBound.TANH:
            return torch.tanh(x)
        elif self.output_bound == OutputBound.SIGMOID:
            return torch.sigmoid(x)
        elif self.output_bound == OutputBound.HARDTANH:
            return F.hardtanh(x, self.clamp_min, self.clamp_max)
        elif self.output_bound == OutputBound.CLAMP:
            return torch.clamp(x, self.clamp_min, self.clamp_max)
        else:
            return x

    def _expand_graph_for_batch(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replicate graph topology for batch processing.

        PyG-style batching: shift edge indices by N for each sample in batch.

        Returns:
            edge_index_batch: (2, B*E)
            edge_attr_batch: (B*E, edge_dim)
        """
        if batch_size == 1:
            return self.edge_index, self.edge_attr_raw

        N = self.num_nodes
        E = self.edge_index.shape[1]

        offsets = torch.arange(batch_size, device=self.edge_index.device).unsqueeze(1) * N  # (B, 1)
        edge_index_batch = self.edge_index.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2, E)
        edge_index_batch = (edge_index_batch + offsets.unsqueeze(1)).reshape(2, -1)  # (2, B*E)

        edge_attr_batch = self.edge_attr_raw.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.edge_attr_raw.shape[1])

        return edge_index_batch, edge_attr_batch

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, N, F) dynamic node features over time
            mask: (N,) bool mask of active nodes (optional)

        Returns:
            dict with:
                "pred": (B, N, C) bounded predictions
                "raw":  (B, N, C) unbounded predictions
        """
        B, T, N, F_in = features.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"

        # ============================================================
        # Identity skip: use last timestep's first out_channels features
        # ============================================================
        if self.config.identity_skip:
            C = self.config.out_channels
            x_skip = features[:, -1, :, :C]  # (B, N, C)

        # ============================================================
        # STEP 1: TEMPORAL ENCODING
        # ============================================================
        # Reshape: (B, T, N, F) -> (B*N, T, F)
        x = features.permute(0, 2, 1, 3).reshape(B * N, T, F_in)

        # Encode temporal sequence
        x = self.temporal_encoder(x)  # (B*N, T, H)

        # Aggregate time: (B*N, T, H) -> (B*N, H)
        x = self.temporal_aggregation(x)  # (B*N, H)

        # ============================================================
        # STEP 2: ADD COORDINATE + STATIC FEATURES
        # ============================================================
        parts = [x]

        if self.coord_embed is not None:
            # Expand coords for batch: (N, coord_dim) -> (B*N, coord_dim)
            coord_emb = self.coord_embed(self.coords)  # (N, coord_embed_dim)
            coord_emb = coord_emb.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
            parts.append(coord_emb)

        if self.node_static_features is not None:
            static_feats = self.node_static_features.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
            parts.append(static_feats)

        x = torch.cat(parts, dim=-1)  # (B*N, H + coord_dim + static_dim)
        x = self.node_proj(x)  # (B*N, hidden_dim)

        # ============================================================
        # STEP 3: SPATIAL PROCESSING (Graph Transformer / Message Passing)
        # ============================================================
        # Expand graph for batch
        edge_index_b, edge_attr_raw_b = self._expand_graph_for_batch(B)

        # Encode edge features
        edge_attr_b = self.edge_encoder(edge_attr_raw_b)  # (B*E, edge_hidden_dim)

        # Apply spatial layers
        x = self.spatial_processor(x, edge_index_b, edge_attr_b)  # (B*N, hidden_dim)

        # ============================================================
        # STEP 4: DECODE
        # ============================================================
        raw = self.decoder(x)  # (B*N, C)
        raw = raw.reshape(B, N, -1)  # (B, N, C)

        # ============================================================
        # STEP 5: IDENTITY SKIP
        # ============================================================
        if self.config.identity_skip:
            raw = raw + x_skip

        # ============================================================
        # STEP 6: OUTPUT BOUNDING
        # ============================================================
        pred = self._apply_output_bound(raw)

        # ============================================================
        # STEP 7: APPLY MASK
        # ============================================================
        if mask is not None:
            mask_3d = mask.unsqueeze(0).unsqueeze(-1).float()  # (1, N, 1)
            raw = raw * mask_3d
            pred = pred * mask_3d

        return {
            "pred": pred,
            "raw": raw,
        }

    def estimate_memory_usage(
        self,
        batch_size: int = 1,
        timesteps: int = 4,
        dtype_size: int = 4,
    ) -> Dict[str, float]:
        """
        Estimate memory usage in MB.
        """
        N = self.num_nodes
        E = self.edge_index.shape[1]
        H = self.config.hidden_dim
        F_in = self.config.in_features
        C = self.config.out_channels
        edge_h = self.config.edge_hidden_dim
        num_spatial = self.config.spatial_layers

        memory = {}

        # Parameters
        total_params = sum(p.numel() for p in self.parameters())
        memory["parameters"] = total_params * dtype_size / (1024 ** 2)

        # Buffers (graph)
        buffer_elements = E * 2 + E * self.edge_attr_raw.shape[1] + N * 3
        memory["graph_buffers"] = buffer_elements * dtype_size / (1024 ** 2)

        # Input
        memory["input"] = batch_size * timesteps * N * F_in * dtype_size / (1024 ** 2)

        # Temporal encoder activations
        memory["temporal_activations"] = batch_size * N * timesteps * H * dtype_size / (1024 ** 2)

        # Spatial activations (per layer)
        spatial_node_act = batch_size * N * H
        spatial_edge_act = batch_size * E * edge_h
        memory["spatial_activations_per_layer"] = (spatial_node_act + spatial_edge_act) * dtype_size / (1024 ** 2)
        memory["spatial_activations_total"] = memory["spatial_activations_per_layer"] * num_spatial

        # Attention matrices (only for transformer mode)
        if self.config.spatial_mode == SpatialMode.TRANSFORMER:
            # TransformerConv computes attention per edge, not N^2 matrix
            memory["attention_per_layer"] = batch_size * E * self.config.num_heads * dtype_size / (1024 ** 2)
            memory["attention_total"] = memory["attention_per_layer"] * num_spatial
        else:
            memory["attention_total"] = 0

        # Output
        memory["output"] = batch_size * N * C * dtype_size / (1024 ** 2)

        # Backward (roughly 2x forward activations)
        forward_act = (
            memory["temporal_activations"]
            + memory["spatial_activations_total"]
            + memory.get("attention_total", 0)
        )
        memory["backward_estimate"] = forward_act * 2

        memory["total_estimate"] = (
            memory["parameters"]
            + memory["graph_buffers"]
            + memory["input"]
            + forward_act
            + memory["backward_estimate"]
            + memory["output"]
        )

        return memory

    @torch.no_grad()
    def get_graph_info(self) -> Dict[str, Union[int, float]]:
        """Return graph topology statistics."""
        E = self.edge_index.shape[1]
        N = self.num_nodes

        row = self.edge_index[0]
        degree = torch.zeros(N, dtype=torch.long, device=self.edge_index.device)
        degree.scatter_add_(0, row, torch.ones_like(row))

        dist = self.edge_attr_raw[:, 3] if self.edge_attr_raw.shape[1] > 3 else torch.norm(
            self.edge_attr_raw[:, :3], dim=-1
        )

        return {
            "num_nodes": N,
            "num_edges": E,
            "avg_degree": E / N,
            "min_degree": degree.min().item(),
            "max_degree": degree.max().item(),
            "mean_degree": degree.float().mean().item(),
            "min_edge_length": dist.min().item(),
            "max_edge_length": dist.max().item(),
            "mean_edge_length": dist.mean().item(),
        }
```
