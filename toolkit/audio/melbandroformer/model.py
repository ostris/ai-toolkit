# Vendored from ZFTurbo/Music-Source-Separation-Training (MIT, Roman Solovyev 2024),
# derived from lucidrains/BS-RoFormer (MIT, Phil Wang 2023). See LICENSE in this directory.
# Inference-only: training loss, PoPE, linear attention and checkpointing removed;
# rotary_embedding_torch inlined; attention is plain torch SDPA; the 60 per-band
# linears are stacked bmm's (upstream per-band checkpoint keys are repacked on load).
# core() is torch.compile-friendly: no einops, no complex ops, dynamic batch dim.
from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from einops import rearrange, pack, unpack, reduce, repeat


def exists(val):
    return val is not None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# rotary (matches rotary_embedding_torch.RotaryEmbedding(dim) defaults: freqs_for='lang', theta=1e4)

def rotate_half(x):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(Module):
    def __init__(self, dim):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        # nn.Parameter (not buffer) so the key matches upstream checkpoints
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def rotate_queries_or_keys(self, t):
        # t: (b h n d); rotation is computed in fp32 regardless of autocast
        n = t.shape[-2]
        with torch.autocast(device_type=t.device.type, enabled=False):
            seq = torch.arange(n, device=t.device, dtype=self.freqs.dtype)
            freqs = torch.einsum('n,f->nf', seq, self.freqs).repeat_interleave(2, dim=-1)
            tf = t.float()
            out = tf * freqs.cos() + rotate_half(tf) * freqs.sin()
        return out.type(t.dtype)


# attention

class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., rotary_embed=None):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm(x)

        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)  # (b h n d)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)

        gates = self.to_gates(x)
        out = out * gates.transpose(1, 2).unsqueeze(-1).sigmoid()

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.,
        ff_dropout=0.,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_embed=rotary_embed),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# band split / mask estimator: all bands in stacked bmm's instead of 60 per-band linears.
# Parameters are stored stacked and zero-padded to the widest band; upstream per-band
# checkpoints are repacked on load (see _load_from_state_dict), so keys stay compatible.

class BandSplit(Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = tuple(dim_inputs)
        n, max_in = len(dim_inputs), max(dim_inputs)
        self.n, self.max_in = n, max_in

        self.gamma = nn.Parameter(torch.ones(n, max_in))
        self.weight = nn.Parameter(torch.empty(n, max_in, dim))
        self.bias = nn.Parameter(torch.zeros(n, dim))
        for i, d in enumerate(dim_inputs):
            bound = d ** -0.5
            nn.init.uniform_(self.weight[i, :d], -bound, bound)
            nn.init.uniform_(self.bias[i], -bound, bound)

        # gather map from the concatenated band features into the padded (n, max_in) layout;
        # pads read the extra zero column appended in forward. Built on cpu for meta-device init.
        pad_src = torch.full((n, max_in), sum(dim_inputs), dtype=torch.long, device='cpu')
        offset = 0
        for i, d in enumerate(dim_inputs):
            pad_src[i, :d] = torch.arange(offset, offset + d, device='cpu')
            offset += d
        self.register_buffer('pad_src', pad_src.flatten(), persistent=False)
        self.register_buffer('scale', torch.tensor([d ** 0.5 for d in dim_inputs], device='cpu')[:, None], persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        if f"{prefix}to_features.0.1.weight" in state_dict:
            ws, bs, gs = [], [], []
            for i, d in enumerate(self.dim_inputs):
                ws.append(F.pad(state_dict.pop(f"{prefix}to_features.{i}.1.weight").t(), (0, 0, 0, self.max_in - d)))
                bs.append(state_dict.pop(f"{prefix}to_features.{i}.1.bias"))
                gs.append(F.pad(state_dict.pop(f"{prefix}to_features.{i}.0.gamma"), (0, self.max_in - d)))
            state_dict[f"{prefix}weight"] = torch.stack(ws)
            state_dict[f"{prefix}bias"] = torch.stack(bs)
            state_dict[f"{prefix}gamma"] = torch.stack(gs)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, x):
        b, t, _ = x.shape
        x = F.pad(x.reshape(b * t, -1), (0, 1))[:, self.pad_src].view(b * t, self.n, self.max_in)
        # per-band RMSNorm; zero pads do not change the norm
        norm = x.float().norm(dim=-1, keepdim=True).clamp(min=1e-12)
        x = (x / norm * self.scale * self.gamma).transpose(0, 1)  # (n, M, max_in)
        out = torch.baddbmm(self.bias[:, None, :], x, self.weight)  # (n, M, dim)
        return out.transpose(0, 1).reshape(b, t, self.n, -1)


class MaskEstimator(Module):
    def __init__(self, dim, dim_inputs, depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = tuple(dim_inputs)
        n, hidden = len(dim_inputs), dim * mlp_expansion_factor
        dim_outs = [2 * d for d in dim_inputs]  # GLU halves it back to d
        max_out = max(dim_outs)
        self.n, self.depth, self.max_out = n, depth, max_out

        dims = (dim, *((hidden,) * depth), max_out)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(n, i, o)) for i, o in zip(dims[:-1], dims[1:])])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(n, o)) for o in dims[1:]])
        for w, bias, i in zip(self.weights, self.biases, dims[:-1]):
            nn.init.uniform_(w, -i ** -0.5, i ** -0.5)
            nn.init.uniform_(bias, -i ** -0.5, i ** -0.5)

        # GLU gather: a * sigmoid(b) with a/b the first/second half of each band's valid output columns
        glu_a, glu_b = [], []
        for i, d in enumerate(dim_inputs):
            base = i * max_out
            glu_a.append(torch.arange(base, base + d, device='cpu'))
            glu_b.append(torch.arange(base + d, base + 2 * d, device='cpu'))
        self.register_buffer('glu_a', torch.cat(glu_a), persistent=False)
        self.register_buffer('glu_b', torch.cat(glu_b), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        if f"{prefix}to_freqs.0.0.0.weight" in state_dict:
            for j in range(self.depth + 1):
                ws, bs = [], []
                for i in range(self.n):
                    w = state_dict.pop(f"{prefix}to_freqs.{i}.0.{2 * j}.weight").t()
                    bias = state_dict.pop(f"{prefix}to_freqs.{i}.0.{2 * j}.bias")
                    if j == self.depth:
                        w = F.pad(w, (0, self.max_out - w.shape[1]))
                        bias = F.pad(bias, (0, self.max_out - bias.shape[0]))
                    ws.append(w)
                    bs.append(bias)
                state_dict[f"{prefix}weights.{j}"] = torch.stack(ws)
                state_dict[f"{prefix}biases.{j}"] = torch.stack(bs)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, x):
        b, t, n, d = x.shape
        h = x.permute(2, 0, 1, 3).reshape(n, b * t, d)
        for j in range(self.depth):
            h = torch.tanh(torch.baddbmm(self.biases[j][:, None, :], h, self.weights[j]))
        out = torch.baddbmm(self.biases[-1][:, None, :], h, self.weights[-1])  # (n, M, max_out)
        out = out.transpose(0, 1).reshape(b * t, n * self.max_out)
        out = out[:, self.glu_a] * torch.sigmoid(out[:, self.glu_b])
        return out.view(b, t, -1)


# main class

class MelBandRoformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        sample_rate=44100,  # for the librosa mel filter bank
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        zero_dc=True,
        mask_estimator_depth=1,
        mlp_expansion_factor=4,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        # one rotary module shared across all time (resp. freq) transformers, as upstream
        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs),
            ]))

        self.stft_window_fn = partial(torch.hann_window, stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        freqs = stft_n_fft // 2 + 1

        # binary mel filter bank as in section 2 of the paper; overlapping bands' masks are averaged
        from librosa import filters  # lazy: ~1s import, only needed here
        mel_filter_bank = torch.from_numpy(filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands))
        # upstream forces the first/last bins in so every frequency is covered by some band
        mel_filter_bank[0][0] = 1.
        mel_filter_bank[-1, -1] = 1.

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

        # explicit cpu so the loader can build the module under torch.device('meta')
        repeated_freq_indices = repeat(torch.arange(freqs, device='cpu'), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2, device='cpu')
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([
            MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            for _ in range(num_stems)
        ])

        self.zero_dc = zero_dc

    def stft(self, raw_audio):
        """raw_audio (b s t) -> band-gathered real features (b t (f c)), and stft_repr (b (f s) t c)."""
        device = raw_audio.device
        batch = raw_audio.shape[0]

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # merge stereo / mono into the frequency dim, frequency leading, for band splitting
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        batch_arange = torch.arange(batch, device=device)[..., None]

        x = stft_repr[batch_arange, self.freq_indices]

        # fold complex into the frequency dim
        x = rearrange(x, 'b f t c -> b t (f c)')
        return x, stft_repr

    def core(self, x):
        """Real-valued trunk (b t (f c)) -> masks (b n t (f c)); the torch.compile target."""
        x = self.band_split(x)

        # axial attention: time then freq

        b, t, f, d = x.shape
        # native views instead of einops: same math, ~3x faster dynamo tracing
        for time_transformer, freq_transformer in self.layers:
            x = time_transformer(x.transpose(1, 2).reshape(b * f, t, d)).view(b, f, t, d)
            x = freq_transformer(x.transpose(1, 2).reshape(b * t, f, d)).view(b, t, f, d)

        return torch.stack([fn(x) for fn in self.mask_estimators], dim=1)

    def apply_masks(self, stft_repr, masks, channels):
        batch = stft_repr.shape[0]
        num_stems = masks.shape[1]

        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)

        # modulate frequency representation (complex multiply)

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)

        masks = masks.type(stft_repr.dtype)

        # average the estimated mask over overlapping bands

        scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=num_stems, t=stft_repr.shape[-1])

        stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        stft_repr = stft_repr * masks_averaged

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        if self.zero_dc:
            stft_repr[:, 0] = 0.

        stft_window = self.stft_window_fn(device=stft_repr.device)
        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio

    def compile_core(self, **kwargs):
        """Fuse the trunk's elementwise chains; batch dim is marked dynamic so any batch size hits one graph."""
        compiled = torch.compile(self.core, **kwargs)

        def core(x):
            torch._dynamo.mark_dynamic(x, 0)
            return compiled(x)

        self.core = core
        return self

    def forward(self, raw_audio):
        """
        raw_audio: (b t) or (b s t). Returns (b s t) for a single stem, else (b n s t).
        b batch, f freq, t time, s audio channel, n stem, c complex (2), d feature
        """
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]

        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), \
            'stereo model needs 2-channel audio, mono model needs 1-channel audio'

        x, stft_repr = self.stft(raw_audio)
        masks = self.core(x)
        return self.apply_masks(stft_repr, masks, channels)
