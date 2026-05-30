from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

try:
    import pywt
except ImportError:
    pywt = None


ComplexCoeffDict = Dict[str, torch.Tensor]
_BankKey = Tuple[str, Tuple[int, ...], torch.device, torch.dtype]

_ALL_SUBBANDS = ("LL", "LH", "HL", "HH")
_SUBBAND_TO_INDEX = {name: idx for idx, name in enumerate(_ALL_SUBBANDS)}
_ANALYSIS_WEIGHT_CACHE: Dict[_BankKey, torch.Tensor] = {}
_SYNTHESIS_WEIGHT_CACHE: Dict[_BankKey, torch.Tensor] = {}
_DUAL_ANALYSIS_WEIGHT_CACHE: Dict[_BankKey, Tuple[torch.Tensor, torch.Tensor]] = {}
_DUAL_SYNTHESIS_WEIGHT_CACHE: Dict[_BankKey, Tuple[torch.Tensor, torch.Tensor]] = {}


class WaveletBackendError(ImportError):
    pass


_SUPPORTED_PADDING_MODES = {
    "zero": "constant",
    "constant": "constant",
    "reflect": "reflect",
    "replicate": "replicate",
    "circular": "circular",
    "periodization": "circular",
    "symmetric": "reflect",
}


# =========================
# Utils
# =========================

def _require_complex(x: torch.Tensor):
    if not torch.is_complex(x):
        raise TypeError("Input must be complex tensor")


def _canonical_padding_mode(mode: str) -> str:
    mode = mode.lower()
    if mode not in _SUPPORTED_PADDING_MODES:
        raise ValueError(f"Unsupported padding mode: {mode}")
    return _SUPPORTED_PADDING_MODES[mode]


def _normalize_subbands(subbands):
    if subbands is None:
        return _ALL_SUBBANDS

    out, seen = [], set()
    for s in subbands:
        s = s.upper()
        if s not in _SUBBAND_TO_INDEX:
            raise ValueError(f"Invalid subband: {s}")
        if s not in seen:
            out.append(s)
            seen.add(s)

    if not out:
        raise ValueError("subbands不能为空")

    return tuple(out)


def _require_pywt():
    if pywt is None:
        raise WaveletBackendError("pip install PyWavelets")


# =========================
# Filters
# =========================

@lru_cache(maxsize=32)
def _get_analysis_filters(wavelet: str):
    _require_pywt()
    w = pywt.Wavelet(wavelet)
    return tuple(w.dec_lo[::-1]), tuple(w.dec_hi[::-1])


@lru_cache(maxsize=32)
def _get_synthesis_filters(wavelet: str):
    _require_pywt()
    w = pywt.Wavelet(wavelet)
    return tuple(w.rec_lo), tuple(w.rec_hi)


@lru_cache(maxsize=32)
def _get_full_analysis_bank(wavelet: str):
    lo_c, hi_c = _get_analysis_filters(wavelet)
    lo = torch.tensor(lo_c, dtype=torch.float32)
    hi = torch.tensor(hi_c, dtype=torch.float32)

    return torch.stack([
        torch.outer(lo, lo),
        torch.outer(lo, hi),
        torch.outer(hi, lo),
        torch.outer(hi, hi),
    ], dim=0).unsqueeze(1)


@lru_cache(maxsize=32)
def _get_full_synthesis_bank(wavelet: str):
    lo_c, hi_c = _get_synthesis_filters(wavelet)
    lo = torch.tensor(lo_c, dtype=torch.float32)
    hi = torch.tensor(hi_c, dtype=torch.float32)

    return torch.stack([
        torch.outer(lo, lo),
        torch.outer(lo, hi),
        torch.outer(hi, lo),
        torch.outer(hi, hi),
    ], dim=0).unsqueeze(1)


# =========================
# Tensor reshape
# =========================

def _to_bchw(x):
    if x.ndim == 2:
        return x[None, None], "hw"
    if x.ndim == 3:
        return x[:, None], "bhw"
    if x.ndim == 4:
        return x, "bchw"
    raise ValueError(f"Invalid shape {x.shape}")


def _restore(x, tag):
    if tag == "hw":
        return x[0, 0]
    if tag == "bhw":
        return x[:, 0]
    return x


# =========================
# Padding
# =========================

def _pad(x, k, mode):
    p = max(k // 2 - 1, 0)
    if p == 0:
        return x
    return F.pad(x, (p, p, p, p), mode=_canonical_padding_mode(mode))


def _crop(x, k):
    p = max(k // 2 - 1, 0)
    if p == 0:
        return x
    return x[..., p:-p, p:-p]


def _subband_indices(subbands):
    return tuple(_SUBBAND_TO_INDEX[s] for s in subbands)


def _stack_coeffs(coeffs: Dict[str, torch.Tensor], subbands, expected_shape=None):
    ys = []
    tag = None
    ref_shape = None

    for s in subbands:
        if s not in coeffs:
            raise KeyError(f"Missing subband: {s}")
        v, current_tag = _to_bchw(coeffs[s])
        if tag is None:
            tag = current_tag
            ref_shape = v.shape
            if expected_shape is not None and ref_shape != expected_shape:
                raise ValueError(f"Expected coeff shape {expected_shape}, but got {ref_shape} for {s}")
        else:
            if current_tag != tag:
                raise ValueError("All coefficient tensors must share the same layout")
            if v.shape != ref_shape:
                raise ValueError(f"All coefficient tensors must share the same shape, got {v.shape} and {ref_shape}")
        ys.append(v)

    if not ys:
        raise ValueError("coeffs不能为空")

    return torch.stack(ys, dim=2), tag, ref_shape


def _get_cached_weights(cache, cache_key, bank):
    weight = cache.get(cache_key)
    if weight is None:
        weight = _expand_bank(bank, cache_key[1][0])
        cache[cache_key] = weight
    return weight


# =========================
# Core DWT
# =========================

def _expand_bank(bank, C):
    k = bank.shape[-1]
    num_subbands = bank.shape[0]
    return bank.unsqueeze(0).expand(C, -1, -1, -1, -1).reshape(C * num_subbands, 1, k, k)


def complex_dwt2d(x, wavelet="db2", mode="reflect", subbands=None):
    _require_complex(x)

    x, tag = _to_bchw(x)
    B, C, _, _ = x.shape

    subbands = _normalize_subbands(subbands)
    idx = _subband_indices(subbands)
    bank_full = _get_full_analysis_bank(wavelet).to(x.device, x.dtype)
    bank = bank_full[list(idx)]

    k = bank.shape[-1]
    cache_key = (wavelet, (C,) + idx, x.device, x.dtype)
    weight = _get_cached_weights(_ANALYSIS_WEIGHT_CACHE, cache_key, bank)

    x = _pad(x, k, mode)
    y = F.conv2d(x, weight, stride=2, groups=C)

    y = y.view(B, C, len(subbands), y.shape[-2], y.shape[-1])

    return {
        name: _restore(y[:, :, i], tag)
        for i, name in enumerate(subbands)
    }


# =========================
# Core IDWT
# =========================

def complex_idwt2d(coeffs: Dict[str, torch.Tensor], wavelet="db2"):
    subbands = _normalize_subbands(coeffs.keys())

    y, tag, ref_shape = _stack_coeffs(coeffs, subbands)
    B, C, _, H, W = y.shape
    device, dtype = y.device, y.dtype

    idx = _subband_indices(subbands)
    bank_full = _get_full_synthesis_bank(wavelet).to(device, dtype)
    bank = bank_full[list(idx)]

    k = bank.shape[-1]
    cache_key = (wavelet, (C,) + idx, device, dtype)
    weight = _get_cached_weights(_SYNTHESIS_WEIGHT_CACHE, cache_key, bank)

    y = y.reshape(B, C * len(subbands), H, W)

    x = F.conv_transpose2d(
        y,
        weight,
        stride=2,
        groups=C,
    )
    cropped_x = _crop(x, k)

    return _restore(cropped_x, tag)


# =========================
# RA Wrapper
# =========================

def _adc_to_batch(x, layout):
    if layout == "DAR":
        return x[:, None]
    if layout == "ARD":
        return x.permute(2, 0, 1)[:, None]
    raise ValueError


def _batch_to_adc(x, layout):
    x = x[:, 0]
    if layout == "DAR":
        return x
    if layout == "ARD":
        return x.permute(1, 2, 0)
    raise ValueError


def complex_dwt2d_ra(adc, input_layout="DAR", wavelet="db2"):
    x = _adc_to_batch(adc, input_layout)
    coeff = complex_dwt2d(x, wavelet)
    return {k: _batch_to_adc(v, input_layout) for k, v in coeff.items()}


def complex_idwt2d_ra(coeffs, input_layout="DAR", wavelet="db2"):
    coeffs_b = {k: _adc_to_batch(v, input_layout) for k, v in coeffs.items()}
    x = complex_idwt2d(coeffs_b, wavelet)
    return _batch_to_adc(x, input_layout)


# =========================
# Approximate DTCWT / IDTCWT
# =========================

@lru_cache(maxsize=32)
def _get_dual_analysis_banks(wavelet: str):
    bank_a = _get_full_analysis_bank(wavelet).squeeze(1)

    lo_c, hi_c = _get_analysis_filters(wavelet)
    lo_b = torch.roll(torch.tensor(lo_c, dtype=torch.float32), shifts=1)
    hi_b = torch.roll(torch.tensor(hi_c, dtype=torch.float32), shifts=1)
    bank_b = torch.stack([
        torch.outer(lo_b, lo_b),
        torch.outer(lo_b, hi_b),
        torch.outer(hi_b, lo_b),
        torch.outer(hi_b, hi_b),
    ], dim=0)

    return bank_a.unsqueeze(1), bank_b.unsqueeze(1)


@lru_cache(maxsize=32)
def _get_dual_synthesis_banks(wavelet: str):
    bank_a = _get_full_synthesis_bank(wavelet).squeeze(1)

    lo_c, hi_c = _get_synthesis_filters(wavelet)
    lo_b = torch.roll(torch.tensor(lo_c, dtype=torch.float32), shifts=1)
    hi_b = torch.roll(torch.tensor(hi_c, dtype=torch.float32), shifts=1)
    bank_b = torch.stack([
        torch.outer(lo_b, lo_b),
        torch.outer(lo_b, hi_b),
        torch.outer(hi_b, lo_b),
        torch.outer(hi_b, hi_b),
    ], dim=0)

    return bank_a.unsqueeze(1), bank_b.unsqueeze(1)


def complex_dtcwt2d(x, wavelet="db2", mode="reflect", subbands=None):
    _require_complex(x)

    x, tag = _to_bchw(x)
    B, C, _, _ = x.shape

    subbands = _normalize_subbands(subbands)
    idx = _subband_indices(subbands)

    bank_a_full, bank_b_full = _get_dual_analysis_banks(wavelet)
    bank_a = bank_a_full[list(idx)].to(x.device, x.real.dtype)
    bank_b = bank_b_full[list(idx)].to(x.device, x.real.dtype)

    k = bank_a.shape[-1]
    cache_key = (wavelet, (C,) + idx, x.device, x.real.dtype)
    cached_weights = _DUAL_ANALYSIS_WEIGHT_CACHE.get(cache_key)
    if cached_weights is None:
        cached_weights = (_expand_bank(bank_a, C), _expand_bank(bank_b, C))
        _DUAL_ANALYSIS_WEIGHT_CACHE[cache_key] = cached_weights
    weight_a, weight_b = cached_weights

    x_pad_real = _pad(x.real, k, mode)
    ya = F.conv2d(x_pad_real, weight_a, stride=2, groups=C)
    yb = F.conv2d(x_pad_real, weight_b, stride=2, groups=C)

    ya = ya.view(B, C, len(subbands), ya.shape[-2], ya.shape[-1])
    yb = yb.view(B, C, len(subbands), yb.shape[-2], yb.shape[-1])
    y = torch.complex(ya, yb)

    return {
        name: _restore(y[:, :, i], tag)
        for i, name in enumerate(subbands)
    }


def complex_idtcwt2d(coeffs: Dict[str, torch.Tensor], wavelet="db2"):
    subbands = _normalize_subbands(coeffs.keys())

    y, tag, ref_shape = _stack_coeffs(coeffs, subbands)
    B, C, _, H, W = y.shape
    device = y.device
    real_dtype = y.real.dtype

    idx = _subband_indices(subbands)
    bank_a_full, bank_b_full = _get_dual_synthesis_banks(wavelet)
    bank_a = bank_a_full[list(idx)].to(device, real_dtype)
    bank_b = bank_b_full[list(idx)].to(device, real_dtype)

    k = bank_a.shape[-1]
    cache_key = (wavelet, (C,) + idx, device, real_dtype)
    cached_weights = _DUAL_SYNTHESIS_WEIGHT_CACHE.get(cache_key)
    if cached_weights is None:
        cached_weights = (_expand_bank(bank_a, C), _expand_bank(bank_b, C))
        _DUAL_SYNTHESIS_WEIGHT_CACHE[cache_key] = cached_weights
    weight_a, weight_b = cached_weights

    ya = y.real.reshape(B, C * len(subbands), H, W)
    yb = y.imag.reshape(B, C * len(subbands), H, W)

    xa = F.conv_transpose2d(ya, weight_a, stride=2, groups=C)
    xb = F.conv_transpose2d(yb, weight_b, stride=2, groups=C)
    x = torch.complex(xa, xb)

    return _restore(_crop(x, k), tag)


def complex_dtcwt2d_ra(adc, input_layout="DAR", wavelet="db2", mode="reflect", subbands=None):
    x = _adc_to_batch(adc, input_layout)
    coeff = complex_dtcwt2d(x, wavelet=wavelet, mode=mode, subbands=subbands)
    return {k: _batch_to_adc(v, input_layout) for k, v in coeff.items()}


def complex_idtcwt2d_ra(coeffs, input_layout="DAR", wavelet="db2"):
    coeffs_b = {k: _adc_to_batch(v, input_layout) for k, v in coeffs.items()}
    x = complex_idtcwt2d(coeffs_b, wavelet=wavelet)
    return _batch_to_adc(x, input_layout)


# =========================
# TEST
# =========================

def main():
    import numpy as np

    adc = np.load("/home/neu/code/Pitt-Radar/radar_numpy_train/0.npy")
    adc = torch.from_numpy(adc).to(torch.complex64)

    if torch.cuda.is_available():
        adc = adc.cuda()

    print("Input:", adc.shape)

    coeffs = complex_dwt2d_ra(adc)

    print("DWT coeff shapes:")
    for k, v in coeffs.items():
        print(f"{k}: {v.shape}")

    recon = complex_idwt2d_ra(coeffs)
    err = torch.mean(torch.abs(recon - adc))

    print("\nDWT reconstructed:", recon.shape)
    print(f"DWT reconstruction error: {err.item():.8e}")

    dtcwt_coeffs = complex_dtcwt2d_ra(adc)

    print("\nDTCWT coeff shapes:")
    for k, v in dtcwt_coeffs.items():
        print(f"{k}: {v.shape}")

    dtcwt_recon = complex_idtcwt2d_ra(dtcwt_coeffs)
    dtcwt_err = torch.mean(torch.abs(dtcwt_recon - adc))

    print("\nDTCWT reconstructed:", dtcwt_recon.shape)
    print(f"DTCWT reconstruction error: {dtcwt_err.item():.8e}")


if __name__ == "__main__":
    main()