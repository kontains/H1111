import argparse
from datetime import datetime
import gc
import json
import random
import os
import re
import time
import math
import copy
from typing import Tuple, Optional, List, Union, Any, Dict
from rich.traceback import install as install_rich_tracebacks
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from transformers import LlamaModel
from tqdm import tqdm
from rich_argparse import RichHelpFormatter
from networks import lora_framepack
from hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from frame_pack import hunyuan
from frame_pack.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked, load_packed_model
from frame_pack.utils import crop_or_pad_yield_mask, resize_and_center_crop, soft_append_bcthw
from frame_pack.bucket_tools import find_nearest_bucket
from frame_pack.clip_vision import hf_clip_vision_encode
from frame_pack.k_diffusion_hunyuan import sample_hunyuan
from dataset import image_video_dataset

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from utils.device_utils import clean_memory_on_device
from base_hv_generate_video import save_images_grid, save_videos_grid, synchronize_device
from base_wan_generate_video import merge_lora_weights
from frame_pack.framepack_utils import load_vae, load_text_encoder1, load_text_encoder2, load_image_encoders
from dataset.image_video_dataset import load_video
from blissful_tuner.blissful_args import add_blissful_args, parse_blissful_args
from blissful_tuner.video_processing_common import save_videos_grid_advanced
from blissful_tuner.latent_preview import LatentPreviewer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    install_rich_tracebacks()
    parser = argparse.ArgumentParser(description="Framepack inference script", formatter_class=RichHelpFormatter)

    # WAN arguments
    # parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument("--is_f1", action="store_true", help="Use the FramePack F1 model specific logic.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    )

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path. Overrides --model_version if specified.")
    parser.add_argument(
        "--model_version", type=str, default="original", choices=["original", "f1"], help="Select the FramePack model version to use ('original' or 'f1'). Ignored if --dit is specified."
    )
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory or path")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory or path")
    parser.add_argument("--image_encoder", type=str, required=True, help="Image Encoder directory or path")
    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="prompt for generation. If `;;;` is used, it will be split into sections. Example: `section_index:prompt` or "
        "`section_index:prompt;;;section_index:prompt;;;...`, section_index can be `0` or `-1` or `0-2`, `-1` means last section, `0-2` means from 0 to 2 (inclusive).",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, default is empty string. should not change.",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_seconds", type=float, default=5.0, help="video length, Default is 5.0 seconds")
    parser.add_argument("--fps", type=int, default=30, help="video fps, Default is 30")
    parser.add_argument("--infer_steps", type=int, default=25, help="number of inference steps, Default is 25")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=str, default=None, help="Seed for evaluation.")
    # parser.add_argument(
    #     "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    # )
    parser.add_argument("--latent_window_size", type=int, default=9, help="latent window size, default is 9. should not change.")
    parser.add_argument(
        "--embedded_cfg_scale", type=float, default=10.0, help="Embeded CFG scale (distilled CFG Scale), default is 10.0"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for classifier free guidance. Default is 1.0, should not change.",
    )
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="CFG Re-scale, default is 0.0. Should not change.")
    # parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="path to image for image2video inference. If `;;;` is used, it will be used as section images. The notation is same as `--prompt`.",
    )
    parser.add_argument("--end_image_path", type=str, default=None, help="path to end image for image2video inference")
    # parser.add_argument(
    #     "--control_path",
    #     type=str,
    #     default=None,
    #     help="path to control video for inference with controlnet. video file or directory with images",
    # )
    # parser.add_argument("--trim_tail_frames", type=int, default=0, help="trim tail N frames from the video before saving")

    # # Flow Matching
    # parser.add_argument(
    #     "--flow_shift",
    #     type=float,
    #     default=None,
    #     help="Shift factor for flow matching schedulers. Default depends on task.",
    # )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    parser.add_argument("--fp8_fast", action="store_true", help="Enable fast FP8 arithmetic (RTX 4XXX+), only for fp8_scaled mode and can degrade quality slightly but offers noticeable speedup")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],  #  "flash2", "flash3",
        help="attention mode",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    parser.add_argument("--bulk_decode", action="store_true", help="decode all frames at once")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap in the model")
    parser.add_argument(
        "--output_type", type=str, default="video", choices=["video", "images", "latent", "both"], help="output type"
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--compile_args",
        nargs=4,
        metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
        default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
        help="Torch.compile settings",
    )

    # New arguments for batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")

    #parser.add_argument("--preview_latent_every", type=int, default=None, help="Preview latent every N sections")
    parser.add_argument("--preview_suffix", type=str, default=None, help="Unique suffix for preview files to avoid conflicts in concurrent runs.")

    # TeaCache arguments
    parser.add_argument("--use_teacache", action="store_true", help="Enable TeaCache for faster generation.")
    parser.add_argument("--teacache_steps", type=int, default=25, help="Number of steps for TeaCache initialization (should match --infer_steps).")
    parser.add_argument("--teacache_thresh", type=float, default=0.15, help="Relative L1 distance threshold for TeaCache skipping.")
    parser.add_argument(
        "--video_path", # New argument
        type=str,
        default=None,
        help="Path to input video for video-to-video inference (mutually exclusive with --image_path for F1 mode).",
    )

    parser = add_blissful_args(parser)
    args = parser.parse_args()
    args = parse_blissful_args(args)

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.prompt is None and not args.from_file and not args.interactive:
        raise ValueError("Either --prompt, --from_file or --interactive must be specified")
    if args.video_path and not args.is_f1:
        logger.warning("--video_path is primarily designed for F1 mode. Behavior in standard mode might be unexpected.")

    return args


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides

    Args:
        line: Prompt line with options

    Returns:
        Dict[str, Any]: Dictionary of argument overrides
    """
    # TODO common function with hv_train_network.line_to_prompt_dict
    parts = line.split(" --")
    prompt = parts[0].strip()

    # Create dictionary of overrides
    overrides = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        # Map options to argument names
        if option == "w":
            overrides["video_size_width"] = int(value)
        elif option == "h":
            overrides["video_size_height"] = int(value)
        elif option == "f":
            overrides["video_seconds"] = float(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        # elif option == "fs":
        #     overrides["flow_shift"] = float(value)
        elif option == "i":
            overrides["image_path"] = value
        elif option == "cn":
            overrides["control_path"] = value
        elif option == "n":
            overrides["negative_prompt"] = value

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Apply overrides to args

    Args:
        args: Original arguments
        overrides: Dictionary of overrides

    Returns:
        argparse.Namespace: New arguments with overrides applied
    """
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "video_size_width":
            args_copy.video_size[1] = value
        elif key == "video_size_height":
            args_copy.video_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int, float]: (height, width, video_seconds)
    """
    height = args.video_size[0]
    width = args.video_size[1]

    video_seconds = args.video_seconds

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    return height, width, video_seconds


# region DiT model


def get_dit_dtype(args: argparse.Namespace) -> torch.dtype:
    dit_dtype = torch.bfloat16
    if args.precision == "fp16":
        dit_dtype = torch.float16
    elif args.precision == "fp32":
        dit_dtype = torch.float32
    return dit_dtype


def load_dit_model(args: argparse.Namespace, device: torch.device) -> HunyuanVideoTransformer3DModelPacked:
    """load DiT model

    Args:
        args: command line arguments
        device: device to use

    Returns:
        HunyuanVideoTransformer3DModelPacked: DiT model
    """
    loading_device = "cpu"
    # Adjust loading device logic based on F1 requirements if necessary
    if args.blocks_to_swap == 0 and not args.fp8_scaled and args.lora_weight is None:
        loading_device = device

    # F1 model expects bfloat16 according to demo
    # However, load_packed_model might handle dtype internally based on checkpoint.
    # Let's keep the call as is for now.
    logger.info(f"Loading DiT model (Class: HunyuanVideoTransformer3DModelPacked) for {'F1' if args.is_f1 else 'Standard'} mode.")
    model = load_packed_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        loading_device=loading_device,
        # Pass fp8_scaled and split_attn if load_packed_model supports them directly
        # fp8_scaled=args.fp8_scaled, # Assuming load_packed_model handles this
        # split_attn=False, # F1 demo doesn't use split_attn
    )
    return model


def optimize_model(model: HunyuanVideoTransformer3DModelPacked, args: argparse.Namespace, device: torch.device) -> None:
    """optimize the model (FP8 conversion, device move etc.)

    Args:
        model: dit model
        args: command line arguments
        device: device to use
    """
    if args.fp8_scaled:
        # load state dict as-is and optimize to fp8
        state_dict = model.state_dict()

        # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
        move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
        state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=args.fp8_fast)  # args.fp8_fast)

        info = model.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"Loaded FP8 optimized weights: {info}")

        if args.blocks_to_swap == 0:
            model.to(device)  # make sure all parameters are on the right device (e.g. RoPE etc.)
    else:
        # simple cast to dit_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if args.fp8:
            target_dtype = torch.float8e4m3fn

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        if target_device is not None and target_dtype is not None:
            model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    if args.compile:
        compile_backend, compile_mode, compile_dynamic, compile_fullgraph = args.compile_args
        logger.info(
            f"Torch Compiling[Backend: {compile_backend}; Mode: {compile_mode}; Dynamic: {compile_dynamic}; Fullgraph: {compile_fullgraph}]"
        )
        torch._dynamo.config.cache_size_limit = 32
        for i in range(len(model.transformer_blocks)):
            model.transformer_blocks[i] = torch.compile(
                model.transformer_blocks[i],
                backend=compile_backend,
                mode=compile_mode,
                dynamic=compile_dynamic.lower() in "true",
                fullgraph=compile_fullgraph.lower() in "true",
            )

    if args.blocks_to_swap > 0:
        logger.info(f"Enable swap {args.blocks_to_swap} blocks to CPU from device: {device}")
        model.enable_block_swap(args.blocks_to_swap, device, supports_backward=False)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        # make sure the model is on the right device
        model.to(device)

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)


# endregion


# fpack_generate_video.py

def decode_latent(
    latent_window_size: int,
    total_latent_sections: int,
    bulk_decode: bool,
    vae: AutoencoderKLCausal3D,
    latent: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    logger.info(f"Decoding video...")
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)  # add batch dimension

    vae.to(device)
    if not bulk_decode:
        # latent_window_size = latent_window_size  # default is 9 - Redundant assignment
        # total_latent_sections = (args.video_seconds * 30) / (latent_window_size * 4) # Calculation moved to save_output
        # total_latent_sections = int(max(round(total_latent_sections), 1)) # Calculation moved to save_output
        num_frames_per_section_decode = latent_window_size * 4 # How many frames VAE ideally outputs per latent window input
        overlap_frames_needed = latent_window_size * 4 - 3 # Overlap needed for soft_append_bcthw

        latents_to_decode = []
        latent_frame_index = 0

        # --- Calculate chunk sizes based on total_latent_sections ---
        # This loop logic seems complex and depends on how total_latent_sections was calculated.
        # Let's simplify the chunking based *directly* on the input latent length.
        # We iterate through the latent tensor, creating overlapping chunks suitable for the VAE.

        # Simplified chunking logic:
        # Assuming VAE processes chunks related to latent_window_size
        # The exact VAE input requirement isn't perfectly clear from this code alone,
        # but the original loop implies variable section sizes based on being the last one.
        # Let's stick to the original loop structure for chunking latents, as modifying it
        # might break VAE decoding assumptions.

        # Original loop to prepare latent chunks (assuming it's correct for VAE)
        num_frames_per_latent_section_approx = latent_window_size * 2 # Approximate latents per section decode
        for i in range(total_latent_sections - 1, -1, -1):
             is_last_section = i == total_latent_sections - 1
             # Original calculation for section latent frames based on whether it's the last section
             section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)

             # Ensure we don't go out of bounds
             section_end_index = min(latent_frame_index + section_latent_frames, latent.shape[2])
             section_start_index = max(0, section_end_index - section_latent_frames) # Adjust start if end was clipped

             if section_start_index >= section_end_index: # Skip if slice is empty
                 continue

             section_latent = latent[:, :, section_start_index:section_end_index, :, :]
             latents_to_decode.append(section_latent)

             # Advance index based on how many *new* frames were theoretically generated
             # This part is tricky without knowing the exact VAE causal mechanism.
             # Let's use the original advance logic.
             generated_latent_frames_advance = (num_frames_per_section_decode + 3) // 4 + (1 if is_last_section else 0) # Original logic for advancing index
             latent_frame_index += generated_latent_frames_advance # Advance based on original calculation


        latents_to_decode = latents_to_decode[::-1]  # reverse the order of latents to decode

        history_pixels = None
        for idx, latent_chunk in enumerate(tqdm(latents_to_decode, desc="Decoding sections")):
            # Decode the current chunk
            current_pixels = hunyuan.vae_decode(latent_chunk, vae).cpu()
            num_current_frames = current_pixels.shape[2]

            if history_pixels is None:
                history_pixels = current_pixels
            else:
                # Check if the current decoded chunk is long enough for the required overlap
                if num_current_frames >= overlap_frames_needed:
                    # If long enough, use the original soft blending
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlap_frames_needed)
                else:
                    # If too short, concatenate directly to avoid assertion error
                    # Log a warning as this might cause a visual discontinuity
                    logger.warning(f"Last decoded chunk is too short ({num_current_frames} frames) for blending (need {overlap_frames_needed}). Concatenating directly.")
                    history_pixels = torch.cat([history_pixels, current_pixels], dim=2)
            clean_memory_on_device(device)
    else:
        # bulk decode
        logger.info(f"Bulk decoding")
        history_pixels = hunyuan.vae_decode(latent, vae).cpu()
    vae.to("cpu")

    if history_pixels is None:
        logger.error("Decoding failed, history_pixels is None.")
        # Return a dummy tensor or raise an error
        return torch.zeros((1, 3, 1, 64, 64), dtype=torch.float32) # Example dummy tensor

    logger.info(f"Decoded. Pixel shape {history_pixels.shape}")
    return history_pixels[0]  # remove batch dimension


def prepare_i2v_inputs(
    args: argparse.Namespace,
    device: torch.device,
    vae: AutoencoderKLCausal3D,
    encoded_context: Optional[Dict] = None,
    encoded_context_n: Optional[Dict] = None,
) -> Tuple[int, int, float, dict, dict, dict, Optional[torch.Tensor], Optional[torch.Tensor]]: # Added Optional Tensor for video latents
    """Prepare inputs for I2V or V2V

    Args:
        args: command line arguments
        device: device to use
        vae: VAE model, used for image/video encoding
        encoded_context: Pre-encoded text context
        encoded_context_n: Pre-encoded negative text context

    Returns:
        Tuple[int, int, float, dict, dict, dict, torch.Tensor | None, torch.Tensor | None]:
            (height, width, video_seconds, context, context_null, context_img, end_latent, input_video_latents)
    """
    input_video_latents: Optional[torch.Tensor] = None
    last_frame_np_for_encoder: Optional[np.ndarray] = None
    section_images_to_encode: Dict[int, str] = {} # Store paths from --image_path
    using_video_for_conditioning = False

    # define parsing function (remains the same)
    def parse_section_strings(input_string: str) -> dict[int, str]:
        # ... (keep existing parse_section_strings implementation) ...
        section_strings = {}
        if not input_string: # Handle empty input string
            # Return empty dict, caller should handle default later if needed
            return {} # Return empty dict instead of {0: ""} initially
            # return {0: ""}
        if ";;;" in input_string:
            split_section_strings = input_string.split(";;;")
            for section_str in split_section_strings:
                if ":" not in section_str:
                    start = end = 0
                    section_str_val = section_str.strip()
                else:
                    index_str, section_str_val = section_str.split(":", 1)
                    index_str = index_str.strip()
                    section_str_val = section_str_val.strip()

                    m = re.match(r"^(-?\d+)(-\d+)?$", index_str)
                    if m:
                        start = int(m.group(1))
                        end = int(m.group(2)[1:]) if m.group(2) is not None else start
                    else:
                        start = end = 0 # Default to 0 if index format is invalid

                for i in range(start, end + 1):
                    section_strings[i] = section_str_val
        else:
             # If no section specifiers, assume section 0
             section_strings[0] = input_string.strip()


        # Ensure section 0 exists if any sections are defined, using a default if necessary
        if section_strings and 0 not in section_strings:
            indices = list(section_strings.keys())
            try: # Prefer first non-negative index's value for section 0
                first_positive_index = min(i for i in indices if i >= 0)
                section_index = first_positive_index
            except ValueError: # Otherwise prefer smallest negative index's value
                 section_index = min(indices) if indices else 0 # Fallback to 0 if empty

            if section_index in section_strings:
                 section_strings[0] = section_strings[section_index]
            elif section_strings: # If section_index wasn't valid, pick first available
                section_strings[0] = next(iter(section_strings.values()))
            # If section_strings was empty initially, 0 won't be added here.

        # If still no section 0 (e.g., empty input or only negative indices specified)
        # And we NEED a section 0 for fallback logic later.
        # Let's add an empty default only if the dict is not empty, otherwise leave it empty.
        if section_strings and 0 not in section_strings:
            section_strings[0] = "" # Add default empty prompt for section 0

        return section_strings


    # prepare image preprocessing function (remains the same)
    def preprocess_image(image_path: str, target_height: int, target_width: int, is_f1: bool):
        # ... (keep existing preprocess_image implementation) ...
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)  # PIL to numpy, HWC

        if is_f1:
            # F1 specific preprocessing: find bucket (fixed 640 res) and resize/crop
            f1_height, f1_width = find_nearest_bucket(image_np.shape[0], image_np.shape[1], resolution=640)
            # logger.info(f"F1 Mode: Using nearest bucket ({f1_height}, {f1_width}) for image preprocessing.")
            image_np = resize_and_center_crop(image_np, target_width=f1_width, target_height=f1_height)
            # Update target_height/width based on F1 bucket for consistency downstream
            processed_height, processed_width = f1_height, f1_width
        else:
            # Original preprocessing
            image_np = image_video_dataset.resize_image_to_bucket(image_np, (target_width, target_height))
            processed_height, processed_width = image_np.shape[0], image_np.shape[1] # Get actual size after resize

        image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0  # -1 to 1.0, HWC
        image_tensor = image_tensor.permute(2, 0, 1)[None, :, None]  # HWC -> CHW -> NCFHW, N=1, C=3, F=1
        return image_tensor, image_np, processed_height, processed_width

    # --- Input Handling Logic ---
    height, width, video_seconds = check_inputs(args)

    # 1. Handle Video Input (takes precedence for history/primary conditioning)
    if args.video_path:
        using_video_for_conditioning = True
        logger.info(f"Preparing video input from: {args.video_path}")
        input_video_frames_list = load_video(video_path=args.video_path, bucket_reso=(width, height))
        input_video_frames_np = np.stack(input_video_frames_list, axis=0) if input_video_frames_list else np.empty((0, height, width, 3), dtype=np.uint8)        
        num_input_frames = input_video_frames_np.shape[0]
        logger.info(f"Loaded video: {num_input_frames} frames, shape {input_video_frames_np.shape}")

        if args.is_f1 and num_input_frames < 19:
             raise ValueError(f"Input video must have at least 19 frames for F1 mode, but found {num_input_frames}.")

        # Use actual video dimensions
        height, width = input_video_frames_np.shape[1], input_video_frames_np.shape[2]
        args.video_size = [height, width]
        logger.info(f"Using video dimensions: {height}x{width}")

        # VAE Encode Video
        chunk_size = 16
        video_latents_list = []
        last_frame_np_for_encoder = input_video_frames_np[-1].copy() # Get last frame for image encoder

        vae.to(device)
        vae_dtype = vae.dtype
        with torch.no_grad():
            for i in tqdm(range(0, num_input_frames, chunk_size), desc="Encoding video frames"):
                chunk_np = input_video_frames_np[i:i+chunk_size]
                chunk_tensor = torch.from_numpy(chunk_np).float() / 127.5 - 1.0
                chunk_tensor = chunk_tensor.permute(0, 3, 1, 2) # B, C, H, W
                chunk_latents_inner = []
                for frame_idx in range(chunk_tensor.shape[0]):
                    frame_tensor_vae = chunk_tensor[frame_idx:frame_idx+1].unsqueeze(2) # 1, C, 1, H, W
                    with torch.autocast(device_type=device.type, dtype=vae_dtype):
                         latent = hunyuan.vae_encode(frame_tensor_vae, vae).cpu()
                         chunk_latents_inner.append(latent)
                if chunk_latents_inner:
                    chunk_latents_stacked = torch.cat(chunk_latents_inner, dim=2)
                    video_latents_list.append(chunk_latents_stacked)

        vae.to("cpu")
        clean_memory_on_device(device)

        if not video_latents_list:
             raise RuntimeError("Failed to encode any video frames.")
        input_video_latents = torch.cat(video_latents_list, dim=2)
        logger.info(f"Encoded video to latents. Shape: {input_video_latents.shape}")
        del input_video_frames_np # Free video frame memory

    # 2. Parse image paths regardless of video input
    section_image_paths = parse_section_strings(args.image_path)

    # 3. Determine the source for Image Encoder
    if not using_video_for_conditioning:
        if not section_image_paths:
             # No video and no image path provided - need a default or error
             raise ValueError("Must provide --image_path or --video_path for generation.")
             # Or load a default placeholder image? For now, error out.
             # default_img_path = "path/to/default.png"
             # logger.warning(f"No image or video provided, using default image: {default_img_path}")
             # section_image_paths[0] = default_img_path

        # Use the image from section 0 (or fallback) for the image encoder
        image_path_for_encoder = section_image_paths.get(0)
        if not image_path_for_encoder and section_image_paths: # Find fallback if 0 missing
             indices = list(section_image_paths.keys())
             try: first_idx = min(i for i in indices if i >= 0)
             except ValueError: first_idx = min(indices) if indices else None
             if first_idx is not None: image_path_for_encoder = section_image_paths[first_idx]

        if not image_path_for_encoder:
             raise ValueError("Could not determine an image path for image encoder conditioning from --image_path.")

        logger.info(f"Using image {image_path_for_encoder} for Image Encoder conditioning.")
        _, last_frame_np_for_encoder, proc_h, proc_w = preprocess_image(image_path_for_encoder, height, width, args.is_f1)

        # Update H/W based on the processed conditioning image if not F1
        if not args.is_f1:
            height, width = proc_h, proc_w
            args.video_size = [height, width]
            logger.info(f"Updated video size based on processed image: {height}x{width}")
    else:
         # Video is used for conditioning, last_frame_np_for_encoder already set
         logger.info("Using last frame of input video for Image Encoder conditioning.")
         # If F1 mode, video size was already set by video dims. No need to adjust further.

    # --- Now process all images specified in --image_path for VAE encoding ---
    section_images_vae = {} # Store VAE-ready tensors
    section_latents_vae = {} # Store encoded latents

    # Need to potentially resize `--image_path` images based on final H/W determined above
    final_height, final_width = height, width
    first_image_processed = False
    for index, image_path in section_image_paths.items():
         # Reprocess ALL images from --image_path using the *final* determined H/W
         # This ensures consistency if the primary conditioning source changed dimensions
         img_tensor, img_np, proc_h, proc_w = preprocess_image(image_path, final_height, final_width, args.is_f1)
         section_images_vae[index] = img_tensor # Store tensor for VAE
         # Check if F1 bucket size needs to override H/W (only applies if V2V wasn't used or if image 0 dictated size)
         if args.is_f1 and not using_video_for_conditioning and not first_image_processed:
             if proc_h != final_height or proc_w != final_width:
                 logger.info(f"F1 Mode (I2V): Overriding video size to {proc_h}x{proc_w} based on first image bucket.")
                 final_height, final_width = proc_h, proc_w
                 height, width = final_height, final_width # Update primary H/W variables
                 args.video_size = [height, width]
             first_image_processed = True


    # --- Text Encoding ---
    n_prompt = args.negative_prompt if args.negative_prompt else ""
    if encoded_context is None or encoded_context_n is None:
        section_prompts = parse_section_strings(args.prompt)
        # Ensure section 0 exists in prompts, default to "" if not present
        if 0 not in section_prompts:
            section_prompts[0] = ""
            logger.info("Using empty default prompt for section 0.")

        tokenizer1, text_encoder1 = load_text_encoder1(args, args.fp8_llm, device)
        tokenizer2, text_encoder2 = load_text_encoder2(args)
        text_encoder2.to(device)
        logger.info(f"Encoding prompts...")
        llama_vecs, llama_attention_masks, clip_l_poolers = {}, {}, {}
        text_encoder_dtype = torch.float8_e4m3fn if args.fp8_llm else torch.float16
        llama_vec_n, clip_l_pooler_n, llama_attention_mask_n = None, None, None

        with torch.autocast(device_type=device.type, dtype=text_encoder_dtype), torch.no_grad():
            for index, prompt in section_prompts.items():
                 current_prompt = prompt if prompt else ""
                 llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(current_prompt, text_encoder1, text_encoder2, tokenizer1, tokenizer2)
                 llama_vec_padded, llama_attention_mask = crop_or_pad_yield_mask(llama_vec.cpu(), length=512)
                 llama_vecs[index] = llama_vec_padded
                 llama_attention_masks[index] = llama_attention_mask
                 clip_l_poolers[index] = clip_l_pooler.cpu()

                 if index == 0 and args.guidance_scale == 1.0: # Use section 0 as fallback for negative
                     llama_vec_n = torch.zeros_like(llama_vec_padded)
                     llama_attention_mask_n = torch.zeros_like(llama_attention_mask)
                     clip_l_pooler_n = torch.zeros_like(clip_l_poolers[0])

        if args.guidance_scale != 1.0:
             with torch.autocast(device_type=device.type, dtype=text_encoder_dtype), torch.no_grad():
                 current_n_prompt = n_prompt if n_prompt else ""
                 llama_vec_n_raw, clip_l_pooler_n_raw = hunyuan.encode_prompt_conds(current_n_prompt, text_encoder1, text_encoder2, tokenizer1, tokenizer2)
                 llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n_raw.cpu(), length=512)
                 clip_l_pooler_n = clip_l_pooler_n_raw.cpu()

        if llama_vec_n is None: # Handle CFG=1.0 case if section 0 wasn't processed
             logger.warning("Negative prompt tensors not generated (likely guidance_scale=1.0). Using zeros.")
             if 0 in llama_vecs:
                 llama_vec_n = torch.zeros_like(llama_vecs[0])
                 llama_attention_mask_n = torch.zeros_like(llama_attention_masks[0])
                 clip_l_pooler_n = torch.zeros_like(clip_l_poolers[0])
             else: # Absolute fallback - should not happen if prompts parsed correctly
                  raise RuntimeError("Cannot create zero negative prompt - no positive prompt for reference.")


        del text_encoder1, text_encoder2, tokenizer1, tokenizer2
        clean_memory_on_device(device)

        encoded_context = {"llama_vecs": llama_vecs, "llama_attention_masks": llama_attention_masks, "clip_l_poolers": clip_l_poolers}
        encoded_context_n = {"llama_vec": llama_vec_n, "llama_attention_mask": llama_attention_mask_n, "clip_l_pooler": clip_l_pooler_n}
    else:
        logger.info("Using pre-encoded text context.")
        llama_vecs = encoded_context["llama_vecs"]
        llama_attention_masks = encoded_context["llama_attention_masks"]
        clip_l_poolers = encoded_context["clip_l_poolers"]
        llama_vec_n = encoded_context_n["llama_vec"]
        llama_attention_mask_n = encoded_context_n["llama_attention_mask"]
        clip_l_pooler_n = encoded_context_n["clip_l_pooler"]
        section_prompts = parse_section_strings(args.prompt) # Reparse prompts
        if 0 not in section_prompts: section_prompts[0] = ""

    # --- Image Encoder Encoding ---
    # `last_frame_np_for_encoder` holds the correct frame (from video or image path)
    if last_frame_np_for_encoder is None:
         raise RuntimeError("Logic error: No image data determined for image encoder.")

    feature_extractor, image_encoder = load_image_encoders(args)
    image_encoder.to(device)
    logger.info(f"Encoding image for primary conditioning with {'SigLIP' if args.is_f1 else 'Image Encoder'}...")
    img_encoder_dtype = image_encoder.dtype
    with torch.autocast(device_type=device.type, dtype=img_encoder_dtype), torch.no_grad():
        image_encoder_output = hf_clip_vision_encode(last_frame_np_for_encoder, feature_extractor, image_encoder)
        primary_hidden_state = image_encoder_output.last_hidden_state.cpu()

    del image_encoder, feature_extractor, last_frame_np_for_encoder
    clean_memory_on_device(device)

    # --- VAE Encoding Section Images ---
    logger.info(f"Encoding section image(s) specified in --image_path to latent space...")
    vae.to(device)
    vae_dtype = vae.dtype
    section_start_latents = {} # Latents from --image_path
    with torch.autocast(device_type=device.type, dtype=vae_dtype), torch.no_grad():
        for index, img_tensor in section_images_vae.items():
            start_latent = hunyuan.vae_encode(img_tensor, vae).cpu()
            section_start_latents[index] = start_latent
        # Encode end image if provided (original logic)
        end_latent = None
        if args.end_image_path is not None:
             end_img_tensor, _, _, _ = preprocess_image(args.end_image_path, final_height, final_width, args.is_f1)
             end_latent = hunyuan.vae_encode(end_img_tensor, vae).cpu()

    vae.to("cpu")
    clean_memory_on_device(device)

    # --- Prepare Conditioning Dictionaries ---
    arg_c = {} # Positive text conditioning
    arg_null = {} # Negative text conditioning
    arg_c_img = {} # Image conditioning (primary + section-specific latents)

    # Ensure section_prompts exists
    if 'section_prompts' not in locals():
        section_prompts = parse_section_strings(args.prompt)
        if 0 not in section_prompts: section_prompts[0] = ""

    # Populate text args
    for index, prompt in section_prompts.items():
        # Find corresponding encoded text data, fallback to section 0
        vec = llama_vecs.get(index, llama_vecs.get(0))
        mask = llama_attention_masks.get(index, llama_attention_masks.get(0))
        pooler = clip_l_poolers.get(index, clip_l_poolers.get(0))
        if vec is None: continue # Skip if no text data found for index or fallback 0

        arg_c_i = {"llama_vec": vec, "llama_attention_mask": mask, "clip_l_pooler": pooler, "prompt": prompt if prompt else ""}
        arg_c[index] = arg_c_i

    # Ensure section 0 exists in arg_c if others do
    if arg_c and 0 not in arg_c:
        first_key = next(iter(arg_c.keys()))
        arg_c[0] = arg_c[first_key]
        arg_c[0]['prompt'] = section_prompts.get(0, "") # Update prompt text for section 0

    # Populate negative text args
    arg_null = {"llama_vec": llama_vec_n, "llama_attention_mask": llama_attention_mask_n, "clip_l_pooler": clip_l_pooler_n, "prompt": n_prompt}

    # Populate image args
    # The primary conditioning (from video end frame or image path 0) uses index 0
    primary_start_latent = None
    if using_video_for_conditioning:
        # Use the last latent frame of the encoded video
        primary_start_latent = input_video_latents[:, :, -1:].cpu() # Shape (1, C, 1, H, W)
    elif 0 in section_start_latents:
         # Use the latent encoded from image_path section 0
         primary_start_latent = section_start_latents[0]
    elif section_start_latents: # Fallback to first available section latent if 0 missing
         first_key = next(iter(section_start_latents.keys()))
         primary_start_latent = section_start_latents[first_key]
    else:
         # This case should be caught earlier (no image/video provided)
         raise RuntimeError("Cannot determine primary start latent for image conditioning.")

    arg_c_img[0] = {
        "image_encoder_last_hidden_state": primary_hidden_state,
        "start_latent": primary_start_latent
    }

    # Add other section latents (from --image_path) if they exist
    for index, latent in section_start_latents.items():
        if index != 0: # Add if not the primary one already added
             # Use the primary hidden state for all sections, only latent changes
             arg_c_img[index] = {
                 "image_encoder_last_hidden_state": primary_hidden_state,
                 "start_latent": latent
             }

    # Final check for minimal context existence
    if not arg_c or not arg_c_img:
        raise ValueError("Failed to prepare conditioning arguments. Check prompts and image/video paths.")
    if 0 not in arg_c or 0 not in arg_c_img:
         raise ValueError("Section 0 conditioning is missing. This is required.")


    return height, width, video_seconds, arg_c, arg_null, arg_c_img, end_latent, input_video_latents


# def setup_scheduler(args: argparse.Namespace, config, device: torch.device) -> Tuple[Any, torch.Tensor]:
#     """setup scheduler for sampling

#     Args:
#         args: command line arguments
#         config: model configuration
#         device: device to use

#     Returns:
#         Tuple[Any, torch.Tensor]: (scheduler, timesteps)
#     """
#     if args.sample_solver == "unipc":
#         scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False)
#         scheduler.set_timesteps(args.infer_steps, device=device, shift=args.flow_shift)
#         timesteps = scheduler.timesteps
#     elif args.sample_solver == "dpm++":
#         scheduler = FlowDPMSolverMultistepScheduler(
#             num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False
#         )
#         sampling_sigmas = get_sampling_sigmas(args.infer_steps, args.flow_shift)
#         timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
#     elif args.sample_solver == "vanilla":
#         scheduler = FlowMatchDiscreteScheduler(num_train_timesteps=config.num_train_timesteps, shift=args.flow_shift)
#         scheduler.set_timesteps(args.infer_steps, device=device)
#         timesteps = scheduler.timesteps

#         # FlowMatchDiscreteScheduler does not support generator argument in step method
#         org_step = scheduler.step

#         def step_wrapper(
#             model_output: torch.Tensor,
#             timestep: Union[int, torch.Tensor],
#             sample: torch.Tensor,
#             return_dict: bool = True,
#             generator=None,
#         ):
#             return org_step(model_output, timestep, sample, return_dict=return_dict)

#         scheduler.step = step_wrapper
#     else:
#         raise NotImplementedError("Unsupported solver.")

#     return scheduler, timesteps


def generate(args: argparse.Namespace, gen_settings: GenerationSettings, shared_models: Optional[Dict] = None) -> Tuple[AutoencoderKLCausal3D, torch.Tensor]: # Return VAE too
    """main function for generation

    Args:
        args: command line arguments
        gen_settings: Generation settings object
        shared_models: dictionary containing pre-loaded models and encoded data

    Returns:
        Tuple[AutoencoderKLCausal3D, torch.Tensor]: vae, generated latent
    """
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    # Ensure seed is integer
    if isinstance(seed, str):
        try:
            seed = int(seed)
        except ValueError:
            logger.warning(f"Invalid seed string: {seed}. Generating random seed.")
            seed = random.randint(0, 2**32 - 1)
    elif not isinstance(seed, int):
        logger.warning(f"Invalid seed type: {type(seed)}. Generating random seed.")
        seed = random.randint(0, 2**32 - 1)

    args.seed = seed  # set seed to args for saving

    vae = None # Initialize VAE
    input_video_latents = None # Initialize video latents
    real_history_latents = None # Initialize to be safe

    # Check if we have shared models
    if shared_models is not None:
        # Use shared models and encoded data
        vae = shared_models.get("vae")
        model = shared_models.get("model")

        if args.video_path: # V2V with shared models
            logger.warning("V2V with shared models is not fully tested for context caching. Preparing inputs.")
            if vae is None: # Load VAE if not in shared
                 vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
            # Prepare inputs, which will give us input_video_latents
            height, width, video_seconds, context, context_null, context_img, end_latent, input_video_latents = prepare_i2v_inputs(args, device, vae)
            # Note: Caching for V2V contexts (video latents themselves) isn't implemented here
        else: # I2V with shared models
            prompt_key = args.prompt if args.prompt else ""
            n_prompt_key = args.negative_prompt if args.negative_prompt else ""
            encoded_context = shared_models.get("encoded_contexts", {}).get(prompt_key)
            encoded_context_n = shared_models.get("encoded_contexts", {}).get(n_prompt_key)

            if encoded_context is None or encoded_context_n is None:
                 logger.info("Cached context not found or incomplete, preparing inputs.")
                 if vae is None:
                     vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
                 height, width, video_seconds, context, context_null, context_img, end_latent, input_video_latents = prepare_i2v_inputs(
                     args, device, vae
                 ) # input_video_latents will be None
            else:
                 logger.info("Using cached context from shared models.")
                 if vae is None:
                      vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
                 height, width, video_seconds, context, context_null, context_img, end_latent, input_video_latents = prepare_i2v_inputs(
                     args, device, vae, encoded_context, encoded_context_n
                 ) # input_video_latents will be None

    else: # No shared models
        # prepare inputs without shared models
        vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
        # Unpack the returned video latents
        height, width, video_seconds, context, context_null, context_img, end_latent, input_video_latents = prepare_i2v_inputs(args, device, vae)

        # load DiT model
        model = load_dit_model(args, device) # Handles F1 class loading implicitly

        # merge LoRA weights
        if args.lora_weight is not None and len(args.lora_weight) > 0:
             logger.info("Merging LoRA weights...")
             try:
                 from base_wan_generate_video import merge_lora_weights # Example import path
                 merge_lora_weights(lora_framepack, model, args, device)
             except ImportError:
                  logger.error("merge_lora_weights function not found. Skipping LoRA merge.")
             except Exception as e:
                  logger.error(f"Error merging LoRA weights: {e}")

             if args.save_merged_model:
                 logger.info(f"Saving merged model to {args.save_merged_model} and exiting.")
                 # save_model(model, args.save_merged_model) # Implement this
                 return None, None # Indicate no generation occurred


        # optimize model: fp8 conversion, block swap etc.
        optimize_model(model, args, device)
        if args.use_teacache:
            logger.info(f"Initializing TeaCache: steps={args.teacache_steps}, threshold={args.teacache_thresh}")
            model.initialize_teacache(
                enable_teacache=True,
                num_steps=args.teacache_steps,
                rel_l1_thresh=args.teacache_thresh
            )
        else:
            logger.info("TeaCache is disabled.")
            model.initialize_teacache(enable_teacache=False)

    # --- Sampling ---
    latent_window_size = args.latent_window_size
    # `video_seconds` from `prepare_i2v_inputs` reflects:
    # - Total duration for I2V
    # - *Additional* duration to generate for V2V
    # This `total_latent_sections_for_duration_arg` is used for I2V or to calculate *new* V2V sections.
    total_latent_sections_for_duration_arg = (video_seconds * args.fps) / (latent_window_size * 4)
    total_latent_sections_for_duration_arg = int(max(round(total_latent_sections_for_duration_arg), 1))

    # set random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    f1_frames_per_section = latent_window_size * 4 - 3

    logger.info(
        f"Mode: {'F1' if args.is_f1 else 'Standard'}, "
        f"Video size: {height}x{width}@{video_seconds:.2f}s ({'Additional duration for V2V' if input_video_latents is not None else 'Total duration for I2V'}), "
        f"fps: {args.fps}, infer_steps: {args.infer_steps}, frames per generation step: {f1_frames_per_section}"
    )

    compute_dtype = model.dtype if hasattr(model, 'dtype') else torch.bfloat16
    if args.fp8 or args.fp8_scaled:
        logger.info("FP8 enabled, using bfloat16 for intermediate computations.")
        compute_dtype = torch.bfloat16
    logger.info(f"Using compute dtype: {compute_dtype}")


    # --- F1 Model Specific Sampling Logic ---
    if args.is_f1:
        logger.info("Starting F1 model sampling process.")
        f1_sampler = 'unipc'
        f1_guidance_scale = 1.0
        f1_embedded_cfg_scale = 10.0
        f1_guidance_rescale = 0.0
        logger.info(f"F1 Mode: Using sampler={f1_sampler}, guidance_scale={f1_guidance_scale}, "
                    f"embedded_cfg_scale={f1_embedded_cfg_scale}, guidance_rescale={f1_guidance_rescale}")

        loop_iterations = 0
        is_v2v = input_video_latents is not None
        total_latent_frames_in_history = 0 # Will store total valid frames in history_latents

        if is_v2v:
            logger.info(f"Initializing F1 history with {input_video_latents.shape[2]} frames from input video.")
            history_latents = input_video_latents.cpu().float() # Shape (1, C, T_video, H, W)
            total_latent_frames_in_history = history_latents.shape[2]
            if total_latent_frames_in_history < 19:
                 raise ValueError(f"Input video resulted in {total_latent_frames_in_history} latent frames. F1 mode requires at least 19 for history initialization.")

            loop_iterations = total_latent_sections_for_duration_arg # This is already for *additional* time
            logger.info(f"Generating {loop_iterations} new sections ({video_seconds:.2f} additional seconds) after the input video.")
        else: # I2V Mode
            logger.info("Initializing F1 history from start image (section 0).")
            history_latents = torch.zeros((1, 16, 19, height // 8, width // 8), dtype=torch.float32, device='cpu')
            start_latent_0 = context_img.get(0, {}).get("start_latent")
            if start_latent_0 is None:
                 raise ValueError("Cannot find primary start_latent (index 0) in context_img for I2V.")
            history_latents = torch.cat([history_latents, start_latent_0.cpu().float()], dim=2) # Shape (1, C, 20, H, W)
            total_latent_frames_in_history = 1 # The single start image latent is the only 'real' history
            loop_iterations = total_latent_sections_for_duration_arg # Use total sections for I2V
            logger.info(f"Generating {loop_iterations} sections from image.")

        if args.preview_latent_every:
            previewer = LatentPreviewer(args, vae, None, gen_settings.device, compute_dtype, model_type="framepack")

        for section_index in range(loop_iterations):
            logger.info(f"--- F1 Section {section_index + 1} / {loop_iterations} ---")

            actual_context_idx = section_index if section_index in context else 0
            actual_image_context_idx = section_index if section_index in context_img else 0
            current_context_data = context.get(actual_context_idx, context.get(0))
            current_image_context_data = context_img.get(actual_image_context_idx, context_img.get(0))

            # Check if data retrieval was successful
            if current_context_data is None:
                logger.error(f"Could not retrieve text context data for section {section_index} (fallback index {actual_context_idx}). Skipping section.")
                # Optionally: Decide whether to skip the section or reuse last section's context
                continue # Skip this section if context is missing
            if current_image_context_data is None:
                logger.error(f"Could not retrieve image context data for section {section_index} (fallback index {actual_image_context_idx}). Using section 0's image context as fallback.")
                # Fallback to section 0 image context if specific one is missing but text exists
                current_image_context_data = context_img.get(0)
                if current_image_context_data is None:
                    logger.error("Fallback to section 0 image context also failed. Skipping section.")
                    continue # Skip if even fallback fails

            current_prompt = current_context_data.get("prompt", "N/A")
            logger.info(f"Using prompt from section {actual_context_idx}: '{current_prompt[:100]}...'") # Log the index actually used
            logger.info(f"Using image context from section {actual_image_context_idx}") # Log the index actually used

            # Use the retrieved data
            llama_vec = current_context_data["llama_vec"].to(device, dtype=compute_dtype)
            llama_attention_mask = current_context_data["llama_attention_mask"].to(device)
            clip_l_pooler = current_context_data["clip_l_pooler"].to(device, dtype=compute_dtype)
            image_encoder_last_hidden_state = current_image_context_data["image_encoder_last_hidden_state"].to(device, dtype=compute_dtype)

            # History latent remains the same calculation based on the overall history
            start_latent_cond = history_latents[:, :, -1:].to(device, dtype=torch.float32)

            llama_vec_n = context_null["llama_vec"].to(device, dtype=compute_dtype)
            llama_attention_mask_n = context_null["llama_attention_mask"].to(device)
            clip_l_pooler_n = context_null["clip_l_pooler"].to(device, dtype=compute_dtype)

            num_new_latents = latent_window_size
            split_sizes = [1, 16, 2, 1, num_new_latents]
            indices = torch.arange(0, sum(split_sizes)).unsqueeze(0).to(device)
            (
                clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices,
                clean_latent_1x_indices, latent_indices
            ) = indices.split(split_sizes, dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            current_history_for_clean = history_latents[:, :, -19:, :, :].to(device, dtype=torch.float32)
            clean_latents_4x, clean_latents_2x, clean_latents_1x = current_history_for_clean.split([16, 2, 1], dim=2)
            clean_latents_input = torch.cat([start_latent_cond, clean_latents_1x], dim=2)

            generated_latents_step = sample_hunyuan(
                transformer=model, sampler=f1_sampler, width=width, height=height, frames=f1_frames_per_section,
                real_guidance_scale=f1_guidance_scale, distilled_guidance_scale=f1_embedded_cfg_scale,
                guidance_rescale=f1_guidance_rescale, num_inference_steps=args.infer_steps, generator=seed_g,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n, device=device, dtype=compute_dtype,
                image_embeddings=image_encoder_last_hidden_state, latent_indices=latent_indices,
                clean_latents=clean_latents_input, clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
            )

            history_latents = torch.cat([history_latents, generated_latents_step.cpu().float()], dim=2)
            total_latent_frames_in_history += int(generated_latents_step.shape[2])

            if args.preview_latent_every is not None and (section_index + 1) % args.preview_latent_every == 0:
                logger.info(f"Previewing latents at section {section_index + 1}")
                preview_latents_full = history_latents.to(gen_settings.device)
                previewer.preview(preview_latents_full, section_index, preview_suffix=args.preview_suffix)
                del preview_latents_full
                clean_memory_on_device(gen_settings.device)

            logger.info(f"Section {section_index + 1} finished. Total latent frames in history: {total_latent_frames_in_history}. History shape: {history_latents.shape}")

            del generated_latents_step, current_history_for_clean, clean_latents_input, clean_latents_1x, clean_latents_2x, clean_latents_4x
            del llama_vec, llama_attention_mask, clip_l_pooler, image_encoder_last_hidden_state, start_latent_cond
            del llama_vec_n, llama_attention_mask_n, clip_l_pooler_n
            clean_memory_on_device(device)

        if is_v2v:
            # For V2V, we need to include some of the original frames to ensure proper overlapping during decoding
            # The minimum overlap needed for decoding is latent_window_size * 4 - 3 frames
            original_frame_count = input_video_latents.shape[2]
            logger.info(f"V2V mode: Original video had {original_frame_count} latent frames, total history has {history_latents.shape[2]} frames")
            
            # Calculate how many original frames we need to keep for proper overlapping
            overlap_needed = latent_window_size * 4 - 3
            # We need to keep at least overlap_needed frames from the original video
            frames_to_keep = min(overlap_needed, original_frame_count)
            
            # Extract the necessary original frames plus all newly generated frames
            start_idx = max(0, original_frame_count - frames_to_keep)
            real_history_latents = history_latents[:, :, start_idx:]
            
            # Calculate how many new frames we actually have
            new_frames_count = history_latents.shape[2] - original_frame_count
            logger.info(f"V2V mode: Keeping {frames_to_keep} frames from original video for overlap + {new_frames_count} newly generated frames")
            logger.info(f"V2V mode: Total frames for decoding: {real_history_latents.shape[2]}")
            
            # Sanity check - if somehow we ended up with too few frames, use more of the original video
            if real_history_latents.shape[2] < overlap_needed and history_latents.shape[2] >= overlap_needed:
                logger.warning(f"V2V slicing resulted in too few frames ({real_history_latents.shape[2]}) for proper overlap. Using more of original video.")
                real_history_latents = history_latents[:, :, -overlap_needed:] # Use at least overlap_needed frames
        else: # I2V - remove initial conditioning zeros
            real_history_latents = history_latents[:, :, 19:]
            # Sanity check frame count
            expected_i2v_frames = 1 + (loop_iterations * (f1_frames_per_section - 3) // 4 if loop_iterations > 0 else 0) # This calculation might be tricky
            # A simpler check is just ensuring it's not empty
            if real_history_latents.shape[2] == 0 and loop_iterations > 0:
                logger.error(f"I2V slicing resulted in empty latents. Original history shape: {history_latents.shape}")
                real_history_latents = history_latents[:, :, -1:] # Fallback to keep at least something

    # --- Standard Model Sampling Logic ---
    elif args.video_path:
         logger.warning("Video input (--video_path) used with standard model. This is experimental and likely won't work correctly without dedicated V2V logic for standard mode.")
         # For standard model with video_path, we should still only return new frames
         # But since we don't have proper V2V implementation for standard model, we'll just return empty latents
         # This will be improved when proper V2V for standard model is implemented
         logger.warning("Standard model V2V not fully implemented - returning empty latents to avoid saving original video")
         real_history_latents = torch.zeros(1, 16, 1, height//8, width//8) # Empty placeholder instead of input_video_latents
    else: # Standard model I2V
        logger.info("Starting standard model sampling process.")
        history_latents_std = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32, device='cpu')
        if end_latent is not None:
            logger.info(f"Using end image: {args.end_image_path}")
            history_latents_std[:, :, 0:1] = end_latent.cpu().float()

        total_generated_latent_frames_std = 0
        loop_iterations_std = total_latent_sections_for_duration_arg # Use the duration arg for std I2V

        latent_paddings = list(reversed(range(loop_iterations_std)))
        if loop_iterations_std > 4:
            logger.info("Using F1-style latent padding heuristic for > 4 sections.")
            latent_paddings = [3] + [2] * (loop_iterations_std - 3) + [1, 0]

        if args.preview_latent_every:
            previewer_std = LatentPreviewer(args, vae, None, gen_settings.device, compute_dtype, model_type="framepack")

        # Temporary accumulator for standard mode
        accumulated_std_latents = []

        for section_index_reverse, latent_padding in enumerate(latent_paddings):
            section_index = loop_iterations_std - 1 - section_index_reverse
            section_index_from_last = -(section_index_reverse + 1)
            logger.info(f"--- Standard Section {section_index + 1} / {loop_iterations_std} (Reverse Index {section_index_reverse}, Padding {latent_padding}) ---")

            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            apply_section_image = False
            image_index = 0 # Default
            if section_index_from_last in context_img:
                image_index = section_index_from_last
                if not is_last_section: apply_section_image = True
            elif section_index in context_img:
                image_index = section_index
                if not is_last_section: apply_section_image = True
            
            start_latent_section = context_img[image_index]["start_latent"].to(device, dtype=torch.float32)
            if apply_section_image:
                latent_padding_size = 0
                logger.info(f"Applying experimental section image, forcing latent_padding_size = 0")

            split_sizes_std = [1, latent_padding_size, latent_window_size, 1, 2, 16]
            indices_std = torch.arange(0, sum(split_sizes_std)).unsqueeze(0).to(device)
            (
                clean_latent_indices_pre, blank_indices, latent_indices_std,
                clean_latent_indices_post, clean_latent_2x_indices_std, clean_latent_4x_indices_std
            ) = indices_std.split(split_sizes_std, dim=1)
            clean_latent_indices_std = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            current_history_std_for_clean = history_latents_std[:, :, :19].to(device, dtype=torch.float32)
            clean_latents_post_std, clean_latents_2x_std, clean_latents_4x_std = current_history_std_for_clean.split([1, 2, 16], dim=2)
            clean_latents_std_input = torch.cat([start_latent_section, clean_latents_post_std], dim=2)

            prompt_index = 0 # Default
            if section_index_from_last in context: prompt_index = section_index_from_last
            elif section_index in context: prompt_index = section_index
            
            context_for_index = context[prompt_index]
            logger.info(f"Using prompt from section {prompt_index}: '{context_for_index['prompt'][:100]}...'")

            llama_vec_std = context_for_index["llama_vec"].to(device, dtype=compute_dtype)
            llama_attention_mask_std = context_for_index["llama_attention_mask"].to(device)
            clip_l_pooler_std = context_for_index["clip_l_pooler"].to(device, dtype=compute_dtype)
            image_encoder_last_hidden_state_std = context_img[image_index]["image_encoder_last_hidden_state"].to(device, dtype=compute_dtype)

            llama_vec_n_std = context_null["llama_vec"].to(device, dtype=compute_dtype)
            llama_attention_mask_n_std = context_null["llama_attention_mask"].to(device)
            clip_l_pooler_n_std = context_null["clip_l_pooler"].to(device, dtype=compute_dtype)

            sampler_to_use = args.sample_solver
            guidance_scale_to_use = args.guidance_scale
            embedded_cfg_scale_to_use = args.embedded_cfg_scale
            guidance_rescale_to_use = args.guidance_rescale

            generated_latents_step_std = sample_hunyuan(
                transformer=model, sampler=sampler_to_use, width=width, height=height, frames=f1_frames_per_section,
                real_guidance_scale=guidance_scale_to_use, distilled_guidance_scale=embedded_cfg_scale_to_use,
                guidance_rescale=guidance_rescale_to_use, num_inference_steps=args.infer_steps, generator=seed_g,
                prompt_embeds=llama_vec_std, prompt_embeds_mask=llama_attention_mask_std, prompt_poolers=clip_l_pooler_std,
                negative_prompt_embeds=llama_vec_n_std, negative_prompt_embeds_mask=llama_attention_mask_n_std,
                negative_prompt_poolers=clip_l_pooler_n_std, device=device, dtype=compute_dtype,
                image_embeddings=image_encoder_last_hidden_state_std, latent_indices=latent_indices_std,
                clean_latents=clean_latents_std_input, clean_latent_indices=clean_latent_indices_std,
                clean_latents_2x=clean_latents_2x_std, clean_latent_2x_indices=clean_latent_2x_indices_std,
                clean_latents_4x=clean_latents_4x_std, clean_latent_4x_indices=clean_latent_4x_indices_std,
            )

            if is_last_section:
                logger.info("Standard Mode: Last section, prepending start latent.")
                generated_latents_step_std = torch.cat([start_latent_section.cpu().float(), generated_latents_step_std.cpu().float()], dim=2)
            else:
                 generated_latents_step_std = generated_latents_step_std.cpu().float()

            # Prepend to list for correct order
            accumulated_std_latents.insert(0, generated_latents_step_std)
            total_generated_latent_frames_std += generated_latents_step_std.shape[2]

            # Update history_latents_std by taking the latest relevant parts from accumulated_std_latents
            # This is a simplification; true history update for standard mode might be more complex
            temp_combined_history = torch.cat(accumulated_std_latents, dim=2)
            if temp_combined_history.shape[2] >= 19:
                 history_latents_std = temp_combined_history[:,:,-19:,:,:] # Keep last 19 frames for next step's conditioning
            else: # Pad if not enough frames yet
                 padding_needed = 19 - temp_combined_history.shape[2]
                 padding = torch.zeros_like(temp_combined_history[:,:,:1]) # Get C,H,W from existing
                 padding = padding.repeat(1,1,padding_needed,1,1)
                 history_latents_std = torch.cat([padding, temp_combined_history], dim=2)


            if args.preview_latent_every is not None and (section_index_reverse + 1) % args.preview_latent_every == 0:
                logger.info(f"Previewing latents at section {section_index + 1} (Reverse Index {section_index_reverse})")
                current_full_video_latents = torch.cat(accumulated_std_latents, dim=2)
                preview_latents_std = current_full_video_latents.to(gen_settings.device)
                previewer_std.preview(preview_latents_std, section_index, preview_suffix=args.preview_suffix)
                del preview_latents_std
                clean_memory_on_device(gen_settings.device)

            logger.info(f"Standard Section {section_index + 1} finished. Frames this step: {generated_latents_step_std.shape[2]}. Total accumulated: {total_generated_latent_frames_std}")

            del generated_latents_step_std, current_history_std_for_clean, clean_latents_std_input, clean_latents_post_std, clean_latents_2x_std, clean_latents_4x_std
            del llama_vec_std, llama_attention_mask_std, clip_l_pooler_std, image_encoder_last_hidden_state_std, start_latent_section
            del llama_vec_n_std, llama_attention_mask_n_std, clip_l_pooler_n_std
            clean_memory_on_device(device)

        # After loop, combine all parts for standard mode
        if accumulated_std_latents:
            real_history_latents = torch.cat(accumulated_std_latents, dim=2)
        else: # Should not happen if loop_iterations_std > 0
             real_history_latents = torch.zeros(1, 16, 1, height//8, width//8)


    # --- End of Sampling Logic ---
    if args.blocks_to_swap > 0 and hasattr(model, 'offloader_double') and model.offloader_double is not None:
        if hasattr(model.offloader_double, 'wait_for_all_submitted_ops'):
            model.offloader_double.wait_for_all_submitted_ops()
        if hasattr(model.offloader_single, 'wait_for_all_submitted_ops'):
            model.offloader_single.wait_for_all_submitted_ops()

    gc.collect()
    clean_memory_on_device(device)

    if real_history_latents is None:
        logger.error("real_history_latents is None before returning from generate(). This should not happen.")
        # Create a dummy tensor to prevent crash, but this indicates a logic flaw.
        real_history_latents = torch.zeros((1, 16, 1, height // 8, width // 8), dtype=compute_dtype, device='cpu')


    logger.info(f"Generation complete. Final latent shape: {real_history_latents.shape}")
    return vae, real_history_latents


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int, original_base_name: Optional[str] = None) -> str: # Add original_base_name
    """Save latent to file

    Args:
        latent: Latent tensor (CTHW expected)
        args: command line arguments
        height: height of frame
        width: width of frame
        original_base_name: Optional base name from loaded file

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}" # Use provided base name
    video_seconds = args.video_seconds
    latent_path = f"{save_path}/{time_flag}_{seed}{original_name}_latent.safetensors" # Add original name to file

    # Ensure latent is on CPU before saving
    latent = latent.detach().cpu()

    if args.no_metadata:
        metadata = None
    else:
        # (Metadata creation remains the same)
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "video_seconds": f"{video_seconds}",
            "infer_steps": f"{args.infer_steps}",
            "guidance_scale": f"{args.guidance_scale}",
            "latent_window_size": f"{args.latent_window_size}",
            "embedded_cfg_scale": f"{args.embedded_cfg_scale}",
            "guidance_rescale": f"{args.guidance_rescale}",
            "sample_solver": f"{args.sample_solver}",
            # "latent_window_size": f"{args.latent_window_size}", # Duplicate key
            "fps": f"{args.fps}",
            "is_f1": f"{args.is_f1}", # Add F1 flag to metadata
        }
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"
        # Add other relevant args like LoRA, compile settings, etc. if desired

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_video(
    video: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None, latent_frames: Optional[int] = None
) -> str:
    """Save video to file

    Args:
        video: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved video file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    latent_frames = "" if latent_frames is None else f"_{latent_frames}"
    video_path = f"{save_path}/{time_flag}_{seed}{original_name}{latent_frames}.mp4"

    video = video.unsqueeze(0)
    if args.codec is not None:
        save_videos_grid_advanced(video, video_path, args.codec, args.container, rescale=True, fps=args.fps, keep_frames=args.keep_pngs)
    else:
        save_videos_grid(video, video_path, fps=args.fps, rescale=True)
    logger.info(f"Video saved to: {video_path}")

    return video_path


def save_images(sample: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    """Save images to directory

    Args:
        sample: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved images directory
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"
    sample = sample.unsqueeze(0)
    save_images_grid(sample, save_path, image_name, rescale=True)
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


# In fpack_generate_video.py

def save_output(
    args: argparse.Namespace,
    vae: AutoencoderKLCausal3D,
    latent: torch.Tensor,
    device: torch.device,
    original_base_names: Optional[List[str]] = None,
) -> None:
    """save output

    Args:
        args: command line arguments
        vae: VAE model
        latent: latent tensor (should be BCTHW or CTHW)
        device: device to use
        original_base_names: original base names (if latents are loaded from files)
    """
    if latent.ndim == 4: # Add batch dim if missing (CTHW -> BCTHW)
        latent = latent.unsqueeze(0)
    elif latent.ndim != 5:
        raise ValueError(f"Unexpected latent dimensions: {latent.ndim}. Expected 4 or 5.")

    # Latent shape is BCTHW
    batch_size, channels, latent_frames, latent_height, latent_width = latent.shape
    height = latent_height * 8
    width = latent_width * 8
    logger.info(f"Saving output. Latent shape: {latent.shape}; Target pixel shape: {height}x{width}")

    actual_latent_frames_count = latent.shape[2] # Assuming latent is BCTHW

    if args.output_type == "latent" or args.output_type == "both":
        # save latent (use first name if multiple originals)
        base_name = original_base_names[0] if original_base_names else None
        save_latent(latent[0], args, height, width, original_base_name=base_name) # Save first batch item if B > 1
    if args.output_type == "latent":
        return

    # Calculate total_latent_sections for decode_latent based on the actual number of frames in the 'latent' tensor.
    # This 'latent' tensor is the complete history (original + generated for V2V).
    LWS = args.latent_window_size
    calculated_total_sections_for_decode: int
    if actual_latent_frames_count <= 0:
        calculated_total_sections_for_decode = 0
    else:
        # The decode_latent loop processes sections, and the 'is_last_section' logic
        # means the first iteration of its loop (which corresponds to the end of the video)
        # effectively advances the latent frame index by LWS + 1. Other iterations advance by LWS.
        adv_first_iter_in_decode_loop = LWS + 1 # Advance for the section where is_last_section=true in generated_latent_frames
        
        calculated_total_sections_for_decode = 1 # At least one section if frames > 0
        remaining_frames = actual_latent_frames_count - adv_first_iter_in_decode_loop
        
        if remaining_frames > 0:
            calculated_total_sections_for_decode += (remaining_frames + LWS - 1) // LWS # Ceil division for other sections

    logger.info(f"Decoding {actual_latent_frames_count} latent frames. Calculated total sections for decode_latent: {calculated_total_sections_for_decode}.")
    total_latent_sections_to_pass_to_decode = calculated_total_sections_for_decode

    # Decode (handle potential batch > 1?)
    # decode_latent expects BCTHW or CTHW, and returns CTHW
    # Currently process only the first item in the batch for saving video/images
    video = decode_latent(args.latent_window_size, total_latent_sections_to_pass_to_decode, args.bulk_decode, vae, latent[0], device)

    if args.output_type == "video" or args.output_type == "both":
        # save video
        original_name = original_base_names[0] if original_base_names else None
        save_video(video, args, original_name, latent_frames=actual_latent_frames_count)

    elif args.output_type == "images":
        # save images
        original_name = original_base_names[0] if original_base_names else None
        save_images(video, args, original_name)


def preprocess_prompts_for_batch(prompt_lines: List[str], base_args: argparse.Namespace) -> List[Dict]:
    """Process multiple prompts for batch mode

    Args:
        prompt_lines: List of prompt lines
        base_args: Base command line arguments

    Returns:
        List[Dict]: List of prompt data dictionaries
    """
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Parse prompt line and create override dictionary
        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = None  # default
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    logger.info(f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}")

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


# In fpack_generate_video.py

def main():
    # Parse arguments
    args = parse_args()

    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device # Ensure args has the final device

    if latents_mode:
        # --- Latent Decode Mode ---
        # (Keep existing logic, but maybe add F1 flag reading from metadata?)
        original_base_names = []
        latents_list = []
        seeds = []
        is_f1_from_metadata = False # Default

        # Allow only one latent file for simplicity now
        if len(args.latent_path) > 1:
             logger.warning("Loading multiple latents is not fully supported for metadata consistency. Using first latent's metadata.")

        for i, latent_path in enumerate(args.latent_path):
             logger.info(f"Loading latent from: {latent_path}")
             base_name = os.path.splitext(os.path.basename(latent_path))[0]
             original_base_names.append(base_name)
             seed = 0 # Default seed

             if not latent_path.lower().endswith(".safetensors"):
                 logger.warning(f"Loading from non-safetensors file {latent_path}. Metadata might be missing.")
                 latents = torch.load(latent_path, map_location="cpu")
                 if isinstance(latents, dict) and "latent" in latents: # Handle potential dict structure
                     latents = latents["latent"]
             else:
                 try:
                     # Load latent tensor
                     loaded_data = load_file(latent_path, device="cpu") # Load to CPU
                     latents = loaded_data["latent"]

                     # Load metadata
                     metadata = {}
                     with safe_open(latent_path, framework="pt", device="cpu") as f:
                         metadata = f.metadata()
                     if metadata is None:
                         metadata = {}
                     logger.info(f"Loaded metadata: {metadata}")

                     # Apply metadata only from the first file for consistency
                     if i == 0:
                         if "seeds" in metadata:
                             try:
                                 seed = int(metadata["seeds"])
                             except ValueError:
                                 logger.warning(f"Could not parse seed from metadata: {metadata['seeds']}")
                         if "height" in metadata and "width" in metadata:
                             try:
                                 height = int(metadata["height"])
                                 width = int(metadata["width"])
                                 args.video_size = [height, width]
                                 logger.info(f"Set video size from metadata: {height}x{width}")
                             except ValueError:
                                 logger.warning(f"Could not parse height/width from metadata.")
                         if "video_seconds" in metadata:
                              try:
                                  args.video_seconds = float(metadata["video_seconds"])
                                  logger.info(f"Set video seconds from metadata: {args.video_seconds}")
                              except ValueError:
                                  logger.warning(f"Could not parse video_seconds from metadata.")
                         if "fps" in metadata:
                             try:
                                 args.fps = int(metadata["fps"])
                                 logger.info(f"Set fps from metadata: {args.fps}")
                             except ValueError:
                                  logger.warning(f"Could not parse fps from metadata.")
                         if "is_f1" in metadata:
                             is_f1_from_metadata = metadata["is_f1"].lower() == 'true'
                             if args.is_f1 != is_f1_from_metadata:
                                  logger.warning(f"Metadata indicates is_f1={is_f1_from_metadata}, overriding command line argument --is_f1={args.is_f1}")
                                  args.is_f1 = is_f1_from_metadata


                 except Exception as e:
                     logger.error(f"Error loading safetensors file {latent_path}: {e}")
                     continue # Skip this file

             # Use seed from first file for all if multiple latents are somehow processed
             if i == 0:
                 args.seed = seed
             seeds.append(seed) # Store all seeds read

             logger.info(f"Loaded latent shape: {latents.shape}")

             if latents.ndim == 5:  # [BCTHW]
                 if latents.shape[0] > 1:
                     logger.warning("Latent file contains batch size > 1. Using only the first item.")
                 latents = latents[0]  # Use first item -> [CTHW]
             elif latents.ndim != 4:
                 logger.error(f"Unexpected latent dimension {latents.ndim} in {latent_path}. Skipping.")
                 continue

             latents_list.append(latents)

        if not latents_list:
             logger.error("No valid latents loaded. Exiting.")
             return

        # Stack latents into a batch if multiple were loaded (BCTHW)
        # Note: Saving output currently only processes the first batch item.
        latent_batch = torch.stack(latents_list, dim=0)

        # Load VAE needed for decoding
        vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device)
        # Call save_output with the batch
        save_output(args, vae, latent_batch, device, original_base_names)

    elif args.from_file:
        # Batch mode from file (Not Implemented)
        logger.error("Batch mode (--from_file) is not implemented yet.")
        # with open(args.from_file, "r", encoding="utf-8") as f:
        #     prompt_lines = f.readlines()
        # prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
        # process_batch_prompts(prompts_data, args) # Needs implementation
        raise NotImplementedError("Batch mode is not implemented yet.")

    elif args.interactive:
        # Interactive mode (Not Implemented)
        logger.error("Interactive mode (--interactive) is not implemented yet.")
        # process_interactive(args) # Needs implementation
        raise NotImplementedError("Interactive mode is not implemented yet.")

    else:
        # --- Single prompt mode (original behavior + F1 support) ---
        gen_settings = get_generation_settings(args)

        # Generate returns (vae, latent)
        vae, latent = generate(args, gen_settings) # VAE might be loaded inside generate

        if latent is None: # Handle cases like --save_merged_model
             logger.info("Generation did not produce latents (e.g., --save_merged_model used). Exiting.")
             return

        # Ensure VAE is available (it should be returned by generate)
        if vae is None:
             logger.error("VAE not available after generation. Cannot save output.")
             return

        # Save output expects BCTHW or CTHW, generate returns BCTHW
        # save_output handles the batch dimension internally now.
        save_output(args, vae, latent, device)

        # Clean up VAE if it was loaded here
        del vae
        gc.collect()
        clean_memory_on_device(device)


    logger.info("Done!")


if __name__ == "__main__":
    main()
