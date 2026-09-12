import copy
import gc
import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch
import folder_paths
import nodes as comfy_nodes
import comfy.model_management
from comfy_extras import nodes_custom_sampler as core_sampler


def _upscale_models():
    category = "latent_upscale_models"
    try:
        names = folder_paths.get_filename_list(category)
    except Exception:
        names = []
    return names or ["minimax_h3_latent_upscaler_3d_fp16.safetensors"]


def _get_viggle_impl():
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("ViggleChunkedSampler")
    if cls is None:
        raise RuntimeError(
            "ViggleChunkedSampler is not loaded. Install/update "
            "Saganaki22/ComfyUI-Viggle-Animate-H3 first."
        )
    mod = importlib.import_module(cls.__module__)
    return cls, mod


_UPSCALE_MOD = None


def _get_upscale_mod():
    global _UPSCALE_MOD
    if _UPSCALE_MOD is not None:
        return _UPSCALE_MOD

    # Prefer an already imported LBH module.
    for name, mod in list(sys.modules.items()):
        if name.endswith("minimax_h3_latent_upscaler_3d") and hasattr(mod, "load_model"):
            _UPSCALE_MOD = mod
            return mod

    comfy_root = Path(folder_paths.__file__).resolve().parent
    candidates = list(
        (comfy_root / "custom_nodes").glob(
            "*Minimax*h3*latent*Upscaler*/nodes/minimax_h3_latent_upscaler_3d.py"
        )
    )
    candidates += list(
        (comfy_root / "custom_nodes").glob(
            "*Minimax*H3*Latent*Upscaler*/nodes/minimax_h3_latent_upscaler_3d.py"
        )
    )

    if not candidates:
        raise RuntimeError(
            "LBH MiniMax H3 Latent Upscaler was not found. "
            "Install LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler."
        )

    path = candidates[0]
    spec = importlib.util.spec_from_file_location(
        "_viggle_lbh_minimax_h3_latent_upscaler_3d", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _UPSCALE_MOD = mod
    return mod


def _learned_upscale_video(video, target_h_px, target_w_px, model_name):
    """Run the exact LBH learned 3D latent upscaler on a plain H3 video latent."""
    mod = _get_upscale_mod()
    src = video
    orig_dtype = src.dtype
    if src.dim() != 5:
        raise ValueError(f"Expected H3 video latent [B,24,T,H,W], got {tuple(src.shape)}")

    dev = mod._resolve_device("cuda")
    compute_dtype = torch.float16
    s = src.to(device=dev, dtype=compute_dtype, copy=True)

    _, _, t, h_in, w_in = s.shape
    downsample = int(getattr(mod, "VAE_DOWNSAMPLE", 16))

    h_out = int(target_h_px // downsample)
    w_out = int(target_w_px // downsample)
    if h_out < h_in or w_out < w_in:
        raise ValueError(
            f"Upscale target {target_w_px}x{target_h_px} is smaller than "
            f"source {w_in * downsample}x{h_in * downsample}."
        )
    if h_out == h_in and w_out == w_in:
        return src.detach().cpu().contiguous()

    effective_scale = (
        target_w_px / (w_in * downsample)
        + target_h_px / (h_in * downsample)
    ) / 2.0

    logging.info(
        "[ViggleChunkedUpscaleRefine] learned upscale %dx%d -> %dx%d, scale %.3f",
        w_in * downsample, h_in * downsample,
        target_w_px, target_h_px, effective_scale,
    )

    model = mod.load_model(model_name, dev, "fp16")
    norm_mean, norm_std = mod._make_norm_tensors(dev, compute_dtype)

    with torch.inference_mode():
        s_norm = (s - norm_mean) / norm_std
        del s
        out = model(
            s_norm,
            scale=effective_scale,
            target_size=(t, h_out, w_out),
            enable_chunking=True,
        )
        del s_norm
        out = out * norm_std + norm_mean

    out = out.to(device="cpu", dtype=orig_dtype, non_blocking=True).contiguous()

    # Match the successful single-shot workflow: release the upscaler before H3 refine.
    if dev.type == "cuda":
        model.to("cpu", non_blocking=True)
        if getattr(mod, "HAS_COMFY_MM", False):
            mod.mm.soft_empty_cache()
        else:
            torch.cuda.empty_cache()
    gc.collect()
    return out


def _same_temporal_plan(a, b):
    keys = ("total_frames", "source_frames", "continuation")
    if any(a.get(k) != b.get(k) for k in keys):
        return False
    return a.get("spans") == b.get("spans")


def _aggressive_vram_cleanup(unload_models=False):
    """Best-effort cleanup between long-video chunks.

    Important: this only releases tensors that no longer have live Python references.
    Chunk outputs/anchors are explicitly moved to CPU before this is called.
    """
    gc.collect()
    try:
        if unload_models:
            fn = getattr(comfy.model_management, "unload_all_models", None)
            if callable(fn):
                fn()
    except Exception as e:
        logging.debug("[ViggleChunkedUpscaleRefine] unload_all_models skipped: %s", e)
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


def _force_offload_vae(vae):
    """Best-effort targeted VAE offload between anchor work and H3 refine.

    The generic unload_all_models() path can leave dynamic models resident.
    For long-video chunking that can make VAE and H3 coexist on GPU and push
    the second high-res refine over the VRAM limit.
    """
    patcher = getattr(vae, "patcher", None)
    if patcher is None:
        return
    offload_device = getattr(patcher, "offload_device", torch.device("cpu"))
    try:
        with torch.inference_mode():
            patcher.unpatch_model(offload_device, unpatch_weights=False)
        logging.info(
            "[ViggleChunkedUpscaleRefine] VAE explicitly offloaded to %s",
            offload_device,
        )
    except Exception as e:
        logging.warning(
            "[ViggleChunkedUpscaleRefine] targeted VAE offload failed: %s", e
        )
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


class ViggleChunkedUpscaleRefineSampler:
    """Long-video Viggle sampler with optional learned H3 latent upscale + per-chunk refine.

    OFF:
        Uses the stock ViggleChunkedSampler with cond_set_direct.

    ON:
        Low-res chunk sample (cond_set_low)
        -> learned LBH H3 3D latent upscale to cond_set_high canvas
        -> second H3/Viggle refine using that chunk's *actual* high-res conditioning
        -> high-res chunk assembly
        -> one final VAE decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "refine_sigmas": ("SIGMAS",),
                "cond_set_direct": ("VIGGLE_COND_SET",),
                "cond_set_low": ("VIGGLE_COND_SET",),
                "cond_set_high": ("VIGGLE_COND_SET",),
                "vae": ("VAE",),
                "use_latent_upscale": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "ON: 0.7MP -> 1.2MP learned latent upscale + refine. OFF: stock 1.0MP direct chunked generation.",
                    },
                ),
                "upscale_model_name": (_upscale_models(),),
                "target_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "rerender_chunk": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64, "step": 1},
                ),
                "rerender_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "chunk_map")
    FUNCTION = "sample"
    CATEGORY = "sampling/viggle"
    DESCRIPTION = (
        "Chunked Viggle sampler with optional 0.7MP -> 1.2MP learned H3 latent "
        "upscale and a second per-chunk refine pass with CPU chunk offload."
    )

    def sample(
        self,
        guider,
        sampler,
        sigmas,
        refine_sigmas,
        cond_set_direct,
        cond_set_low,
        cond_set_high,
        vae,
        use_latent_upscale,
        upscale_model_name,
        target_megapixels,
        seed,
        rerender_chunk,
        rerender_seed,
        dynprompt=None,
        unique_id=None,
    ):
        stock_cls, viggle = _get_viggle_impl()
        stock = stock_cls()

        if not use_latent_upscale:
            frames, chunk_map = stock.sample(
                guider,
                sampler,
                sigmas,
                cond_set_direct,
                vae,
                seed,
                rerender_chunk,
                rerender_seed,
                dynprompt,
                unique_id,
            )
            return (frames, "[Latent Upscale Refine: OFF]\n" + chunk_map)

        if hasattr(viggle, "_validate_sigmas"):
            viggle._validate_sigmas(sigmas)
            viggle._validate_sigmas(refine_sigmas)

        if not _same_temporal_plan(cond_set_low, cond_set_high):
            raise ValueError(
                "Low/high Windowed Conditioning temporal plans must match exactly."
            )

        low_conds = cond_set_low["conds"]
        high_conds = cond_set_high["conds"]
        windows = cond_set_low["windows"]
        source_frames = int(cond_set_low["source_frames"])
        total_frames = int(cond_set_low["total_frames"])
        continuation = cond_set_low.get("continuation", "five_frame_anchor")
        anchor_mode = continuation == "five_frame_anchor"

        if len(low_conds) != len(windows) or len(high_conds) != len(windows):
            raise ValueError("Conditioning/window count mismatch.")

        low_w = int(cond_set_low["width"])
        low_h = int(cond_set_low["height"])
        high_w = int(cond_set_high["width"])
        high_h = int(cond_set_high["height"])
        actual_mp = high_w * high_h / 1_000_000.0

        if high_w < low_w or high_h < low_h:
            raise ValueError("High-resolution conditioning canvas is smaller than low pass.")

        dev = comfy.model_management.intermediate_device()
        storage_device = torch.device("cpu")
        total_latent_t = (total_frames - 1) // viggle.LATENT_TEMPORAL + 1
        total_a = round(total_frames / viggle.FPS * 40)

        if anchor_mode:
            master_low_v = None
            master_low_a = None
        else:
            master_low_v = torch.zeros(
                [1, 24, total_latent_t, low_h // 16, low_w // 16],
                device=storage_device,
            )
            master_low_a = torch.zeros(
                [1, 32, 2, total_a],
                device=storage_device,
            )

        master_high_v = torch.zeros(
            [1, 24, total_latent_t, high_h // 16, high_w // 16],
            device=storage_device,
        )

        noise = core_sampler.Noise_RandomNoise(int(seed))
        low_anchor = None
        high_anchor = None
        low_prev_end = None
        high_prev_end = None

        lines = [
            "[Latent Upscale Refine: ON]",
            "%d chunks, %d source frames; low %dx%d -> high %dx%d (%.3f MP)"
            % (
                len(windows),
                source_frames,
                low_w,
                low_h,
                high_w,
                high_h,
                actual_mp,
            ),
        ]

        pbar = None
        try:
            import comfy.utils as comfy_utils
            pbar = comfy_utils.ProgressBar(len(windows) * 2)
        except Exception:
            pass

        for i, (a, b, lat0, latn) in enumerate(windows):
            seed_i = (
                int(rerender_seed)
                if int(rerender_chunk) == i + 1
                else int(seed) + i
            ) % (1 << 64)

            _aggressive_vram_cleanup(unload_models=False)

            if anchor_mode:
                low_anchor_gpu = None if low_anchor is None else low_anchor.to(dev)
                carry_low = 0 if low_anchor_gpu is None else low_anchor_gpu.shape[2]
                v = torch.zeros(
                    [1, 24, latn, low_h // 16, low_w // 16], device=dev
                )
                a0 = round(a / viggle.FPS * 40)
                a1 = round((b + 1) / viggle.FPS * 40)
                au = torch.zeros([1, 32, 2, a1 - a0], device=dev)
                if low_anchor_gpu is not None:
                    v[:, :, :carry_low] = low_anchor_gpu.to(v)
                    del low_anchor_gpu
            else:
                carry_low = (
                    0
                    if low_prev_end is None
                    else max(0, low_prev_end - lat0)
                )
                v = master_low_v[:, :, lat0 : lat0 + latn].to(dev, non_blocking=True).clone()
                a0 = min(round(a / viggle.FPS * 40), total_a)
                a1 = min(round((b + 1) / viggle.FPS * 40), total_a)
                au = master_low_a[..., a0:a1].to(dev, non_blocking=True).clone()

            out_low_v, out_low_a = stock._sample_window(
                noise,
                guider,
                sampler,
                sigmas,
                low_conds[i],
                seed_i,
                low_h,
                low_w,
                a,
                b,
                carry_low,
                v,
                au,
            )

            del v, au

            low_audio_cpu = out_low_a.detach().to("cpu").contiguous()
            del out_low_a

            if not anchor_mode:
                local_start = max(0, (low_prev_end or 0) - lat0)
                master_low_v[
                    :, :, lat0 + local_start : lat0 + latn
                ] = out_low_v[:, :, local_start:].detach().to("cpu")
                master_low_a[..., a0:a1] = low_audio_cpu

            if i + 1 < len(windows) and anchor_mode:
                next_low_anchor = viggle._encode_anchor(
                    vae, out_low_v, windows[i + 1][0] - a
                )
                low_anchor = next_low_anchor.detach().to("cpu").contiguous()
                del next_low_anchor
                _force_offload_vae(vae)
            else:
                low_anchor = None if i + 1 >= len(windows) else low_anchor

            low_prev_end = lat0 + latn
            if pbar:
                pbar.update(1)

            low_video_cpu = out_low_v.detach().to("cpu").contiguous()
            del out_low_v
            _aggressive_vram_cleanup(unload_models=False)

            up_v_cpu = _learned_upscale_video(
                low_video_cpu,
                high_h,
                high_w,
                upscale_model_name,
            )
            del low_video_cpu

            if up_v_cpu.shape[2] != latn:
                raise ValueError(
                    f"Upscaler changed temporal latent length "
                    f"{latn} -> {up_v_cpu.shape[2]}."
                )

            _aggressive_vram_cleanup(unload_models=False)
            up_v = up_v_cpu.to(dev, non_blocking=True)
            del up_v_cpu

            if anchor_mode:
                high_anchor_gpu = None if high_anchor is None else high_anchor.to(dev)
                carry_high = 0 if high_anchor_gpu is None else high_anchor_gpu.shape[2]
                if high_anchor_gpu is not None:
                    up_v[:, :, :carry_high] = high_anchor_gpu.to(up_v)
                    del high_anchor_gpu
            else:
                carry_high = (
                    0
                    if high_prev_end is None
                    else max(0, high_prev_end - lat0)
                )
                if carry_high:
                    up_v[:, :, :carry_high] = master_high_v[
                        :, :, lat0 : lat0 + carry_high
                    ].to(up_v, non_blocking=True)

            up_v = up_v.to(dev, non_blocking=True)
            refine_au = low_audio_cpu.to(dev, non_blocking=True)

            if up_v.device != refine_au.device:
                raise RuntimeError(
                    f"Internal device mismatch before refine: "
                    f"video={up_v.device}, audio={refine_au.device}"
                )

            _force_offload_vae(vae)
            _aggressive_vram_cleanup(unload_models=False)

            out_high_v, _ = stock._sample_window(
                noise,
                guider,
                sampler,
                refine_sigmas,
                high_conds[i],
                seed_i,
                high_h,
                high_w,
                a,
                b,
                carry_high,
                up_v,
                refine_au,
            )

            del up_v, refine_au, low_audio_cpu

            if anchor_mode:
                local_start = max(0, (high_prev_end or 0) - lat0)
            else:
                local_start = 0
            master_high_v[
                :, :, lat0 + local_start : lat0 + latn
            ] = out_high_v[:, :, local_start:].detach().to("cpu")

            if i + 1 < len(windows) and anchor_mode:
                next_high_anchor = viggle._encode_anchor(
                    vae, out_high_v, windows[i + 1][0] - a
                )
                high_anchor = next_high_anchor.detach().to("cpu").contiguous()
                del next_high_anchor
                _force_offload_vae(vae)
            else:
                high_anchor = None if i + 1 >= len(windows) else high_anchor

            high_prev_end = lat0 + latn

            lines.append(
                "#%d frames %d-%d seed %d | low carry %d | refine carry %d"
                % (i + 1, a, b, seed_i, carry_low, carry_high)
            )
            if pbar:
                pbar.update(1)

            del out_high_v
            _aggressive_vram_cleanup(unload_models=True)

        _aggressive_vram_cleanup(unload_models=True)
        frames = vae.decode(master_high_v)
        if frames.dim() == 5:
            frames = frames.reshape(
                -1,
                frames.shape[-3],
                frames.shape[-2],
                frames.shape[-1],
            )

        if frames.shape[0] < source_frames:
            raise ValueError(
                "Final refined decode is shorter than the source video."
            )
        frames = frames[:source_frames]

        return (frames, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "ViggleChunkedUpscaleRefineSampler": ViggleChunkedUpscaleRefineSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ViggleChunkedUpscaleRefineSampler":
        "Viggle Chunked Sampler + Latent Upscale Refine",
}
