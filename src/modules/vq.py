import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.distributed as dist
from typing import List
from math import ceil
from einops import rearrange, pack, unpack

def first(it):
    return it[0]


def exists(v):
    return v is not None


def identity(t):
    return t


def default(v, d):
    return v if exists(v) else d


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


def get_maybe_sync_seed(device, max_size=10_00):
    rand_int = torch.randint(0, max_size, (), device=device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()


def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        (out,) = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse


def world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_distributed():
    return world_size() > 1


def broadcast_tensors(tensors: List[torch.Tensor], src_rank=0):
    if not is_distributed():
        return
    for tensor in tensors:
        dist.broadcast(tensor, src=src_rank)


def all_reduce_tensors(tensors: List[torch.Tensor], op):
    if not is_distributed():
        return
    for tensor in tensors:
        dist.all_reduce(tensor, op=op)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, epsilon: float = 1e-6):
    return (x + epsilon) / (x.sum() + epsilon * len(x))


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]

def safe_div(num, den, eps = 1e-6):
    return num / den.clamp(min = eps)

def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = l2norm(u + q, dim = 1).detach()

    out = (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    )

    return rearrange(out, '... 1 d -> ... d')

def rotate_to(src, tgt):
    # rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    src, inverse = pack_one(src, '* d')
    tgt, _ = pack_one(tgt, '* d')

    norm_src = src.norm(dim = -1, keepdim = True)
    norm_tgt = tgt.norm(dim = -1, keepdim = True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src),
        safe_div(tgt, norm_tgt),
        src
    )

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()

    return inverse(rotated)
# @torch.no_grad()
# def kmeans(samples: torch.Tensor, nums_clusters: int, kmeans_iters: int):
#     samples = rearrange(samples, "... d -> (...) d")
#     dim, dtype = samples.shape[1], samples.dtype
#     if samples.shape[0] < nums_clusters:
#         random_noise = torch.randn(
#             size=(nums_clusters - samples.shape[0], dim),
#             device=samples.device,
#             dtype=dtype,
#         )
#         samples = torch.cat([samples, random_noise], dim=0)
#     centers = sample_vectors(samples, nums_clusters)
#     for i in range(kmeans_iters):
#         diffs = ((samples.unsqueeze(1) - centers) ** 2).sum(dim=-1)
#         buckets = diffs.argmin(dim=-1)
#         bins = torch.bincount(buckets, minlength=nums_clusters)
#         zero_mask = bins == 0
#         bins[zero_mask] = 1

#         new_centers = centers.new_zeros(nums_clusters, dim, dtype=dtype)
#         new_centers.scatter_add_(0, buckets.unsqueeze(-1).repeat([1, dim]), samples)
#         new_centers = new_centers / bins[..., None]
#         centers = torch.where(zero_mask[..., None], centers, new_centers)

#     return centers, bins


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        embed = uniform_init(codebook_size, dim)

        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.dead_code_threshold = 0.1 / codebook_size
        self.reset_code_threshold = 0.2 / codebook_size
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.inited = False

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

        modified_cluster_size = torch.where(
            mask, self.cluster_size.sum() * self.reset_code_threshold, self.cluster_size
        )
        self.cluster_size.copy_(modified_cluster_size)
        self.embed_avg.copy_(
            self.embed
            * (
                (
                    laplace_smoothing(self.cluster_size, self.epsilon)
                    * self.cluster_size.sum()
                ).unsqueeze_(1)
            )
        )

    def expire_codes_(self, batch_samples):
        expired_codes = (
            self.cluster_size / self.cluster_size.sum() < self.dead_code_threshold
        )
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    # flatten in
    @torch.no_grad()
    def quantize(self, x):
        dist = torch.cdist(x.float(), self.embed.float())
        embed_ind = dist.argmin(dim=-1)
        return embed_ind  # n

    # any in any out
    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    # any in any out
    def encode(self, x):
        shape = x.shape
        # pre-process
        x = rearrange(x, "... d -> (...) d")
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    # equals dequantize
    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x: torch.Tensor, decay: float):

        if not self.inited and self.training:
            broadcast_tensors(self.buffers(), src_rank=0)
            self.inited = True
            
        shape, dtype = x.shape, x.dtype
        x = rearrange(x, "... d -> (...) d")
        if self.training:
            self.expire_codes_(x)

        embed_ind = self.quantize(x)  # BQW
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)  # BQW code
        embed_ind = embed_ind.view(*shape[:-1])  # B Q W
        quantize = self.dequantize(embed_ind).type(dtype)  # B Q W D

        if self.training:
            # 统计的是每一条编码使用过多少次（未归一化），更新
            one_hot_sum = embed_onehot.sum(0)  # codebook   sum up=BQW
            all_reduce_tensors(
                [one_hot_sum], op=dist.ReduceOp.AVG
            )  # 先求和没问题，取平均只是单纯为了数值稳定
            ema_inplace(self.cluster_size, one_hot_sum, decay)
            # 将每条编码对应的embedding全部加起来（未归一化）,更新
            embed_sum = embed_onehot.t() @ x  # codebook dim
            embed_sum = embed_sum.to(torch.float32)
            all_reduce_tensors(
                [embed_sum], op=dist.ReduceOp.AVG
            )  # 先求和没问题，取平均只是单纯为了数值稳定
            ema_inplace(self.embed_avg, embed_sum, decay)
            # 进行一次平滑
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.epsilon)
                * self.cluster_size.sum()
            )
            # 将新的embed替换

            # (3/2x+3/2y) a(a+s+d)
            # 3+a3

            embed_normalized = self.embed_avg * (1.0 / cluster_size.unsqueeze_(1))
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int = None,
        epsilon: float = 1e-6,
        rotation_trick: bool = True,
    ):
        super().__init__()
        self.rotation_trick = rotation_trick
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim, bias=False)
            if requires_projection
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim, bias=False)
            if requires_projection
            else nn.Identity()
        )

        self.epsilon = epsilon

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            epsilon=epsilon,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    # any in any out
    def encode(self, x):
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    # any in any out
    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        return quantize

    def forward(self, x: torch.Tensor, decay: float = 0.99):
        # Input x :(B, C, T)
        input_dtype = x.dtype
        x = x.permute(0, 2, 1) # (B, T, C)

        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x, decay=decay)

        if self.training:
            if self.rotation_trick:
                quantize = rotate_to(x, quantize).to(input_dtype)
            else:
                quantize = x + (quantize - x).detach()

        loss = (
            F.mse_loss(quantize.float(), x.detach().float())
            + F.mse_loss(x.float(), quantize.detach().float()) * 0.25
        )
        quantize = self.project_out(quantize)
        quantize = quantize.permute(0, 2, 1)

        return quantize, embed_ind, loss


class RVQ(nn.Module):
    def __init__(
        self,
        dim,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([])
        for _ in range(num_quantizers):
            self.layers.append(
                VectorQuantization(
                    dim=dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                )
            )
        # quantize dropout
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index

    @property
    def codebook_size(self):
        return first(self.layers).codebook_size

    @property
    def codebook_dim(self):
        return first(self.layers).codebook_dim

    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks)
        return codebooks

    def forward(self, x: torch.Tensor, decay: float = 0.99):
        B, C, T = x.shape
        num_quant = self.num_quantizers
        device = x.device

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:

            # check if seed is manually passed in

            rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(
                self.quantize_dropout_cutoff_index, num_quant
            )

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(
                        rand_quantize_dropout_index + 1, quant_dropout_multiple_of
                    )
                    - 1
                )

        # save all inputs across layers, for use during expiration at end under shared codebook setting

        # go through the layers
        for quantizer_index, vq in enumerate(self.layers):

            if (
                should_quantize_dropout
                and quantizer_index > rand_quantize_dropout_index
            ):
                continue

            # sim vq forward

            quantized, indices, loss = vq(residual, decay=decay)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_losses.append(loss)
            all_indices.append(indices)

        # stack all losses and indices
        all_losses = torch.stack(all_losses, dim=-1)
        all_indices = torch.stack(all_indices, dim=-1)  # B T Nq
        all_indices = all_indices.permute(0,2,1) # B Nq T
        return quantized_out, all_indices, all_losses.mean()

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """
        x: B C T
        """
        assert not self.training
        quantized_out = 0.0
        residual = x
        all_indices = []

        for quantizer_index, vq in enumerate(self.layers):
            quantized, indices, _ = vq(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
        all_indices = torch.stack(all_indices, dim=-1) # (B, T, Nq)
        return all_indices
    
    @torch.no_grad()
    def decode(self, indices: torch.Tensor):
        """
        indices: B T Nq
        """
        quantized_out = 0.0
        quantized_per_layer = []
        # iterate over last dimension of indices
        for i, vq in enumerate(self.layers):
            if i >= indices.shape[-1]: break
            ind = indices[..., i] # (B, T)
            quantized = vq.decode(ind) # (B, C, T)
            quantized_out = quantized_out + quantized
            quantized_per_layer.append(quantized)

        return quantized_out, torch.stack(quantized_per_layer, dim=-1).to(device=indices.device)

if __name__ == "__main__":
    from utils import seed_everything

    seed_everything(42)
    model = RVQ(dim=4, codebook_dim=4, codebook_size=1024, num_quantizers=2)
    # optimizer=torch.optim.AdamW(params=model.parameters(),lr=1e-4)
    for i in range(100):
        x = torch.rand((2000, 4))
        out, _, com_loss = model(x)
        loss = torch.nn.functional.mse_loss(out, x)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        print(loss.item())
