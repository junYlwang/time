"""
Finite Scalar Quantization (FSQ)
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.nn import Module
from torch import tensor, int32
from torch.amp import autocast
from einops import rearrange, pack, unpack

# --- Helper Functions ---

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def identity(t):
    return t

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# --- Rounding & Sampling Logic ---

def round_ste(z):
    """ Standard deterministic rounding with straight-through gradients. """
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    """ Floor with straight through gradients. """
    zhat = z.floor()
    return z + (zhat - z).detach()

class FSQ(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first = False,
        projection_has_bias = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry = False,
        noise_dropout = 0.,
    ):
        super().__init__()

        # Compatibility checks
        assert not (any([l == 2 for l in levels]) and not preserve_symmetry), 'turn on `preserve_symmetry` for levels == 2'

        if isinstance(levels, tuple):
            levels = list(levels)

        _levels = tensor(levels, dtype = int32)
        self.register_buffer('_levels', _levels, persistent = False)

        _basis = torch.cumprod(tensor([1] + levels[:-1]), dim = 0, dtype = int32)
        self.register_buffer('_basis', _basis, persistent = False)

        self.scale = scale
        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)
        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections
        self.return_indices = return_indices

        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer('implicit_codebook', implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32
        self.saved_usage_probs = None

    def _discretize(self, z): #for preserve_symetric=False
        max_level = self._levels.max()
        if self.preserve_symmetry:
            grid = torch.arange(0, max_level, device=z.device).float()
        else:
            max_grid = (max_level - 1) // 2
            min_grid = - (max_level // 2)
            grid = torch.arange(min_grid, max_grid + 1, device=z.device).float()

        z_expanded = z.unsqueeze(-1)
        grid_expanded = grid.view(*([1] * z.ndim), -1)
        dists = (z_expanded - grid_expanded).pow(2)
        logits = -dists

        if self.preserve_symmetry:
            upper_bound = (self._levels - 1).view(1, 1, -1, 1)
            lower_bound = torch.zeros_like(upper_bound)
        else:
            upper_bound = ((self._levels - 1) // 2).view(1, 1, -1, 1)
            lower_bound = (-(self._levels // 2)).view(1, 1, -1, 1)

        mask = (grid_expanded >= lower_bound) & (grid_expanded <= upper_bound)
        logits = logits.masked_fill(~mask, -1e9)
        level_probs = torch.softmax(logits, dim=-1)
        valid_level_probs = []
        for d in range(self.codebook_dim):
            valid = mask[0, 0, 0, d]
            valid_level_probs.append(level_probs[..., d, :][..., valid])
        usage_probs = valid_level_probs[0]
        for probs in valid_level_probs[1:]:
            usage_probs = usage_probs.unsqueeze(-1) * probs.unsqueeze(-2)
            usage_probs = usage_probs.flatten(-2)
        self.saved_usage_probs = usage_probs

        return floor_ste(z + 0.5)

    # --- Bounds ---

    def bound(self, z, eps = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        # Discretize.
        quantized_z = self._discretize(bounded_z)
        half_width = self._levels // 2
        return quantized_z / half_width

    def symmetry_preserving_bound(self, z):
        """ QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1 """

        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        
        # Transform to integer-like space where floor() creates valid levels
        bracket = levels_minus_1 * (z.tanh() + 1) / 2.
        
        # Discretize.
        quantized_bracket = self._discretize(bracket)
        
        return scale * quantized_bracket - 1.

    # --- Quantize ---

    def quantize(self, z):
        shape, device, noise_dropout, preserve_symmetry = z.shape[0], z.device, self.noise_dropout, self.preserve_symmetry
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        # Perform bounding + discretization.
        bounded_z = bound_fn(z)

        # Standard noise dropout.
        if not self.training or noise_dropout == 0.:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    # --- Helpers for Conversion ---

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices):
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat):
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim = -1).round().to(int32)

    def forward(self, z):
        """
        Returns: 
          out: Quantized embedding
          indices: Indices
        
        """
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        z = self.project_in(z)
        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            # --- Quantization happens here. ---
            codes = self.quantize(z)

            indices = None
            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')
            codes = codes.to(orig_dtype)

        out = self.project_out(codes)

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        return out, indices


class RFSQ(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        dim: int | None = None,
        num_quantizers: int = 2,
        channel_first = False,
        projection_has_bias = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry = False,
        noise_dropout = 0.,
    ):
        super().__init__()
        assert num_quantizers >= 1, 'num_quantizers must be at least 1'

        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([
            FSQ(
                levels=levels,
                dim=dim,
                channel_first=channel_first,
                projection_has_bias=projection_has_bias,
                return_indices=return_indices,
                force_quantization_f32=force_quantization_f32,
                preserve_symmetry=preserve_symmetry,
                noise_dropout=noise_dropout,
            )
            for _ in range(num_quantizers)
        ])

    @property
    def codebook_size(self):
        return self.layers[0].codebook_size

    def forward(self, x):
        quantized_out = torch.zeros_like(x)
        residual = x
        all_indices = []
        all_usage_probs = []

        for layer in self.layers:
            quantized, indices = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if exists(indices):
                all_indices.append(indices)
            if exists(layer.saved_usage_probs):
                all_usage_probs.append(layer.saved_usage_probs)

        indices = None
        if all_indices:
            indices = torch.stack(all_indices, dim=1)

        self.saved_usage_probs = all_usage_probs if all_usage_probs else None
        return quantized_out, indices

