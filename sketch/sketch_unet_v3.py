import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sketchutils as util
from typing import Optional, List, Type, Tuple, Dict
from einops import rearrange, repeat

class ImprovedFourierEncoder(nn.Module):
    """
    Enhanced Fourier encoder with better frequency handling
    Based on SD v3 improvements
    """
    def __init__(self, dim: int, max_freq_log2: int = 10):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.max_freq_log2 = max_freq_log2
        
        # Learnable frequency weights with better initialization
        self.freq_weights = nn.Parameter(torch.randn(1, self.half_dim) * 0.02)
        
        # Additional projection for better time representation
        self.projection = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        
        # Enhanced frequency encoding
        freqs = t * self.freq_weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        
        # Concatenate and scale
        embed = torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)
        
        # Additional projection for better representation
        embed = self.projection(embed)
        
        return embed

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for better feature interaction
    """
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, seq_len, query_dim)
        - context: (bs, context_len, context_dim)
        """
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class ImprovedResidualLayer(nn.Module):
    """
    Enhanced residual layer with cross-attention and better normalization
    """
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int, 
                 use_attention: bool = False, attention_heads: int = 8):
        super().__init__()
        
        # Enhanced normalization
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        
        # Improved convolution blocks with better initialization
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
        
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
        
        # Enhanced time adapter with residual connection
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(time_embed_dim * 2, channels)
        )
        
        # Enhanced y adapter with residual connection
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(y_embed_dim * 2, channels)
        )
        
        # Cross-attention for spatial features
        self.use_attention = use_attention
        if use_attention:
            self.cross_attention = CrossAttention(
                query_dim=channels,
                context_dim=channels,
                heads=attention_heads,
                dim_head=channels // attention_heads
            )
            self.attention_norm = nn.GroupNorm(32, channels)
        
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        residual = x

        # First block with normalization
        x = self.norm1(x)
        x = self.block1(x)
        
        # Add time embedding
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + t_embed

        # Add y embedding
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + y_embed

        # Cross-attention if enabled
        if self.use_attention:
            # Reshape for attention: (bs, c, h, w) -> (bs, h*w, c)
            b, c, h, w = x.shape
            x_flat = x.view(b, c, h * w).transpose(1, 2)  # (bs, h*w, c)
            
            # Apply cross-attention
            x_attended = self.cross_attention(x_flat, x_flat)
            x_attended = x_attended.transpose(1, 2).view(b, c, h, w)  # (bs, c, h, w)
            
            # Add with normalization
            x = self.attention_norm(x + x_attended)

        # Second block with normalization
        x = self.norm2(x)
        x = self.block2(x)

        # Add residual connection
        x = x + residual

        return x

class ImprovedEncoder(nn.Module):
    """
    Enhanced encoder with better feature extraction
    """
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, 
                 t_embed_dim: int, y_embed_dim: int, use_attention: bool = False):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ImprovedResidualLayer(channels_in, t_embed_dim, y_embed_dim, use_attention) 
            for _ in range(num_residual_layers)
        ])
        
        # Enhanced downsampling with better initialization
        self.downsample = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, channels_out),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample
        x = self.downsample(x)

        return x

class ImprovedMidcoder(nn.Module):
    """
    Enhanced midcoder with attention
    """
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ImprovedResidualLayer(channels, t_embed_dim, y_embed_dim, use_attention=True) 
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
            
        return x

class ImprovedDecoder(nn.Module):
    """
    Enhanced decoder with better upsampling
    """
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, 
                 t_embed_dim: int, y_embed_dim: int, use_attention: bool = False):
        super().__init__()
        
        # Enhanced upsampling
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels_out),
            nn.SiLU()
        )
        
        self.res_blocks = nn.ModuleList([
            ImprovedResidualLayer(channels_out, t_embed_dim, y_embed_dim, use_attention) 
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Upsample
        x = self.upsample(x)
        
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x

class SketchUNetV3(util.ConditionalVectorField):
    """
    Enhanced SketchUNet with SD v3 improvements
    """
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int): 
        super().__init__()
        
        # Enhanced initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels[0]),
            nn.SiLU()
        )

        # Enhanced time embedder
        self.time_embedder = ImprovedFourierEncoder(t_embed_dim)

        # Enhanced y embedder with better initialization
        self.y_embedder = nn.Embedding(num_embeddings=51, embedding_dim=y_embed_dim)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        
        # Additional projection for y embeddings
        self.y_projection = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(y_embed_dim * 2, y_embed_dim)
        )

        # Encoders, Midcoders, and Decoders with attention
        encoders = []
        decoders = []
        for i, (curr_c, next_c) in enumerate(zip(channels[:-1], channels[1:])):
            # Use attention in deeper layers
            use_attention = i >= len(channels) // 2
            encoders.append(ImprovedEncoder(curr_c, next_c, num_residual_layers, 
                                          t_embed_dim, y_embed_dim, use_attention))
            decoders.append(ImprovedDecoder(next_c, curr_c, num_residual_layers, 
                                          t_embed_dim, y_embed_dim, use_attention))
        
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = ImprovedMidcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
            
        # Enhanced final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, 1, 32, 32)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 1, 32, 32)
        """
        # Enhanced embeddings
        t_embed = self.time_embedder(t) # (bs, time_embed_dim)
        y = y.to(self.y_embedder.weight.device)
        y_embed = self.y_embedder(y) # (bs, y_embed_dim)
        y_embed = self.y_projection(y_embed) # Enhanced y embedding
        
        # Initial convolution
        x = self.init_conv(x) # (bs, c_0, 32, 32)

        residuals = []
        
        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders with skip connections
        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res  # Skip connection
            x = decoder(x, t_embed, y_embed)

        # Final convolution
        x = self.final_conv(x) # (bs, 1, 32, 32)

        return x
