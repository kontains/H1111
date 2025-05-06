import gradio as gr
from gradio import update as gr_update
import subprocess
import threading
import time
import re
import os
import random
import tiktoken
import sys
import ffmpeg
from typing import List, Tuple, Optional, Generator, Dict, Any
import json
from gradio import themes
from gradio.themes.utils import colors
import subprocess
from PIL import Image
import math
import cv2
import glob
import shutil
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
from diffusers_helper.bucket_tools import find_nearest_bucket
import time


# Add global stop event
stop_event = threading.Event()

logger = logging.getLogger(__name__)

def set_random_seed():
    """Returns -1 to set the seed input to random."""
    return -1

def get_step_from_preview_path(path): # Helper function
    # Extracts step number from preview filenames like latent_preview_step_005.mp4
    # or for framepack: latent_preview_section_002.mp4 (assuming sections for framepack)
    # Let's adjust for potential FramePack naming convention (using 'section' instead of 'step')
    base = os.path.basename(path)
    match_step = re.search(r"step_(\d+)", base)
    if match_step:
        return int(match_step.group(1))
    match_section = re.search(r"section_(\d+)", base) # Check for FramePack section naming
    if match_section:
        # Maybe treat sections differently? Or just return the number? Let's return number.
        return int(match_section.group(1))
    return -1 # Default if no number found

def process_framepack_video(
    # --- Standard initial args ---
    prompt: str,
    negative_prompt: str,
    input_image: str, # Start image path
    input_video: Optional[str], # <<< Added Video Path Argument
    # --- End Frame Args ---
    input_end_frame: Optional[str], # End image path
    end_frame_influence: str,
    end_frame_weight: float,
    # --- Model Paths ---
    transformer_path: str,
    vae_path: str,
    text_encoder_path: str,
    text_encoder_2_path: str,
    image_encoder_path: str,
    # --- Core Generation Params ---
    target_resolution: Optional[int],
    framepack_width: Optional[int],
    framepack_height: Optional[int],
    original_dims_str: str,
    total_second_length: float,
    fps: int,
    seed: int,
    steps: int,
    distilled_guidance_scale: float,
    cfg: float,
    rs: float,
    sample_solver: str,
    latent_window_size: int,
    # --- Performance/Memory ---
    fp8: bool,
    fp8_scaled: bool,
    fp8_llm: bool,
    blocks_to_swap: int,
    bulk_decode: bool,
    attn_mode: str,
    vae_chunk_size: Optional[int],
    vae_spatial_tile_sample_min_size: Optional[int],
    device: Optional[str],
    use_teacache: bool,
    teacache_steps: int,
    teacache_thresh: float,
    # --- Batching & Saving ---
    batch_size: int,
    save_path: str, # Argument being checked
    # --- LoRA Params ---
    lora_folder: str,
    enable_preview: bool,
    preview_every_n_sections: int,
    is_f1: bool,
    *args: Any
) -> Generator[Tuple[List[Tuple[str, str]], Optional[str], str, str], None, None]: # Modified return type for preview path
    """Generate video using fpack_generate_video.py"""
    global stop_event
    stop_event.clear()

    # --- Fallback for empty save_path ---
    if not save_path or not save_path.strip():
        print("Warning: save_path was empty, defaulting to 'outputs'")
        save_path = "outputs"
    # --- End Fallback ---

    num_section_controls = 4
    num_loras = 4
    # Calculate slice indices based on the number of components for each group
    secs_end = num_section_controls
    prompts_end = secs_end + num_section_controls
    images_end = prompts_end + num_section_controls
    lora_weights_end = images_end + num_loras
    lora_mults_end = lora_weights_end + num_loras

    # Slice the args tuple to get the values for each group
    framepack_secs = args[0:secs_end]
    framepack_sec_prompts = args[secs_end:prompts_end]
    framepack_sec_images = args[prompts_end:images_end]
    lora_weights_list = list(args[images_end:lora_weights_end]) # Convert tuple slice to list
    lora_multipliers_list = list(args[lora_weights_end:lora_mults_end]) # Convert tuple slice to list

    if not input_image and not any(img for img in framepack_sec_images if img):
        # Yield empty preview path on error
        yield [], None, "Error: Input start image or at least one section image override is required.", ""
        return

    # --- Prepare Section Control Strings ---
    section_prompts_parts = []
    section_images_parts = []

    index_pattern = re.compile(r"^\d+(-\d+)?$")

    for idx_str, sec_prompt, sec_image in zip(framepack_secs, framepack_sec_prompts, framepack_sec_images):
        # Validate the index string format
        if not idx_str or not isinstance(idx_str, str) or not index_pattern.match(idx_str.strip()):
             if idx_str and idx_str.strip(): # Only warn if there was actual input
                 print(f"Warning: Invalid section index/range format '{idx_str}'. Skipping.")
             continue # Skip this entry if the index format is invalid

        current_idx_str = idx_str.strip() # Use the validated string directly

        # Check prompt validity (index string is already validated)
        if sec_prompt and sec_prompt.strip():
            section_prompts_parts.append(f"{current_idx_str}:{sec_prompt.strip()}") # <<< Uses string index/range

        # Check image path validity (index string is already validated)
        if sec_image and os.path.exists(sec_image):
             section_images_parts.append(f"{current_idx_str}:{sec_image}")
    # --- End Section Control String Preparation ---

    final_prompt_arg = prompt # Default to base prompt
    if section_prompts_parts:
        final_prompt_arg = ";;;".join(section_prompts_parts)
        print(f"Using section prompt overrides: {final_prompt_arg}")

    # Determine the image path argument based on section image overrides
    final_image_path_arg = None # Initialize to None
    if section_images_parts:
        final_image_path_arg = ";;;".join(section_images_parts)
        print(f"Using section image overrides for --image_path: {final_image_path_arg}")
    elif input_image: # Only use base input_image if no section overrides are present
        final_image_path_arg = input_image
        print(f"Using base input image for --image_path: {final_image_path_arg}")

    # --- Resolution Calculation ---
    final_height, final_width = None, None
    # Prioritize explicit width/height if valid and divisible by 8
    # (Note: Backend script fpack_generate_video.py checks divisibility by 8,
    #  but the UI logic used 32 based on bucket assumptions. Sticking to 8 here
    #  to match the backend script's direct requirement.)
    if framepack_width is not None and framepack_width > 0 and framepack_height is not None and framepack_height > 0:
        if framepack_width % 8 != 0 or framepack_height % 8 != 0:
             yield [], "Error: Explicit Width and Height must be divisible by 8.", ""
             return
        final_height = int(framepack_height)
        final_width = int(framepack_width)
        print(f"Using explicit dimensions (divisible by 8): H={final_height}, W={final_width}")
    # Fallback to target resolution using bucket logic
    elif target_resolution is not None and target_resolution > 0:
         if not original_dims_str:
              yield [], "Error: Cannot use Target Resolution without an input image to determine aspect ratio.", ""
              return
         try:
             orig_w, orig_h = map(int, original_dims_str.split('x'))
             if orig_w <= 0 or orig_h <= 0:
                 yield [], "Error: Invalid original dimensions stored.", ""
                 return

             # Use find_nearest_bucket (ensure divisible by 8, though buckets often use 32/64)
             # Let's stick to the 32 divisibility from find_nearest_bucket for aspect ratio calc
             bucket_dims = find_nearest_bucket(orig_h, orig_w, resolution=target_resolution) # Bucket logic usually uses 32/64

             if bucket_dims:
                 # Bucket dimensions should already be divisible by 32 (and thus 8)
                 final_height, final_width = bucket_dims
                 print(f"Using Target Resolution {target_resolution}. Found nearest bucket: H={final_height}, W={final_width}")
             else:
                 yield [], f"Error: Could not find a suitable bucket for Target Resolution {target_resolution} and input image aspect ratio.", ""
                 return

         except Exception as e:
             yield [], f"Error calculating bucket dimensions: {e}", ""
             return
    else:
        yield [], "Error: Resolution required. Please provide Target Resolution OR both valid Width and Height (divisible by 8).", ""
        return

    # --- Batch Loop (Simulated for UI) ---
    all_videos = []
    # Calculate total sections for display (doesn't affect backend's internal logic)
    total_sections_estimate_float = (total_second_length * fps) / (latent_window_size * 4) # Use fixed latent_window_size
    total_sections_estimate = int(max(round(total_sections_estimate_float), 1))
    progress_text = f"Starting FramePack generation batch ({total_sections_estimate} estimated sections per video)..."
    status_text = "Preparing batch..."
    yield all_videos, None, status_text, progress_text

    # --- LoRA Setup ---
    valid_loras_paths = []
    valid_loras_mults = []
    if lora_folder and os.path.exists(lora_folder):
        # Use the lora_weights_list and lora_multipliers_list reconstructed from *args
        for weight_name, mult in zip(lora_weights_list, lora_multipliers_list):
            if weight_name and weight_name != "None":
                 # Handle potential full paths or just filenames from dropdown
                 if os.path.isabs(weight_name):
                     lora_path = weight_name
                 else:
                     lora_path = os.path.join(lora_folder, weight_name)

                 if os.path.exists(lora_path):
                     valid_loras_paths.append(lora_path)
                     # Ensure multiplier is converted to string for command line
                     valid_loras_mults.append(str(mult))
                 else:
                     print(f"Warning: LoRA file not found: {lora_path}")
    # Define preview file path base name (backend script creates .mp4 or .png)
    preview_base_path = os.path.join(save_path, "latent_preview")
    preview_mp4_path = preview_base_path + ".mp4"
    preview_png_path = preview_base_path + ".png"

    # --- Loop for UI Batching ---
    for i in range(batch_size):
        if stop_event.is_set():
            yield all_videos, None, "Generation stopped by user.", ""
            return
        
        run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        unique_preview_suffix = f"fpack_{run_id}" # Add prefix for clarity
        # --- Construct unique preview paths ---
        preview_base_path = os.path.join(save_path, f"latent_preview_{unique_preview_suffix}")
        preview_mp4_path = preview_base_path + ".mp4"
        preview_png_path = preview_base_path + ".png"

        current_seed = seed
        if seed == -1: current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1: current_seed = seed + i

        status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed})"
        progress_text = f"Item {i+1}/{batch_size}: Preparing subprocess..."
        current_video_path = None
        # --- Reset preview state for this batch item ---
        current_preview_yield_path = None
        last_preview_mtime = 0
        # --- End Reset ---
        # Initial yield for the new item starting (with empty preview)
        yield all_videos.copy(), current_preview_yield_path, status_text, progress_text

        # --- Prepare Environment and Command ---
        env = os.environ.copy()
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        clear_cuda_cache()

        # --- Command Construction for fpack_generate_video.py ---
        command = [
            sys.executable, "fpack_generate_video.py",
            "--text_encoder1", text_encoder_path, "--text_encoder2", text_encoder_2_path,
            "--image_encoder", image_encoder_path,
            *(["--image_path", final_image_path_arg] if final_image_path_arg else []),
            "--save_path", save_path, "--prompt", final_prompt_arg,
            "--video_size", str(final_height), str(final_width),
            "--video_seconds", str(total_second_length), "--fps", str(fps),
            "--infer_steps", str(steps), "--seed", str(current_seed),
            "--embedded_cfg_scale", str(distilled_guidance_scale),
            "--guidance_scale", str(cfg), "--guidance_rescale", str(rs),
            "--latent_window_size", str(latent_window_size),
            "--sample_solver", sample_solver, "--output_type", "video", "--attn_mode", attn_mode
        ]

        if input_video and os.path.exists(input_video):
             command.extend(["--video_path", input_video])
             print(f"DEBUG: Using input video for V2V: {input_video}")
        if is_f1: command.append("--is_f1")      
        if transformer_path and os.path.exists(transformer_path): command.extend(["--dit", transformer_path.strip()])
        if vae_path and os.path.exists(vae_path): command.extend(["--vae", vae_path.strip()])
        if negative_prompt and negative_prompt.strip(): command.extend(["--negative_prompt", negative_prompt.strip()])
        if input_end_frame and os.path.exists(input_end_frame): command.extend(["--end_image_path", input_end_frame])
        if fp8: command.append("--fp8")
        if fp8 and fp8_scaled: command.append("--fp8_scaled")
        if fp8_llm: command.append("--fp8_llm")
        if bulk_decode: command.append("--bulk_decode")
        if blocks_to_swap > 0: command.extend(["--blocks_to_swap", str(blocks_to_swap)])
        if vae_chunk_size is not None and vae_chunk_size > 0: command.extend(["--vae_chunk_size", str(vae_chunk_size)])
        if vae_spatial_tile_sample_min_size is not None and vae_spatial_tile_sample_min_size > 0: command.extend(["--vae_spatial_tile_sample_min_size", str(vae_spatial_tile_sample_min_size)])
        if device and device.strip(): command.extend(["--device", device.strip()])
        if valid_loras_paths:
            command.extend(["--lora_weight"] + valid_loras_paths)
            command.extend(["--lora_multiplier"] + valid_loras_mults)
        if enable_preview and preview_every_n_sections > 0:
            command.extend(["--preview_latent_every", str(preview_every_n_sections)])
            # --- ADDED: Pass the unique suffix ---
            command.extend(["--preview_suffix", unique_preview_suffix])
            # --- End Pass Suffix ---
            print(f"DEBUG: Enabling preview every {preview_every_n_sections} sections with suffix {unique_preview_suffix}.")
        if use_teacache:
            command.append("--use_teacache")
            command.extend(["--teacache_steps", str(teacache_steps)])
            command.extend(["--teacache_thresh", str(teacache_thresh)])            

        # Ensure all command parts are strings
        command_str = [str(c) for c in command]
        print(f"Running FramePack Command: {' '.join(command_str)}")

        # --- Execute Subprocess & Monitor ---
        p = subprocess.Popen(
            command_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, encoding='utf-8', errors='replace', bufsize=1
        )
        current_phase = "Preparing"
        # --- State variables for tracking actual progress ---
        actual_total_sections = None # Will store the dynamically detected total
        display_section_num = 1 # Initialize display at 1, will be updated
        # --- End State variables ---

        while True:
            if stop_event.is_set():
                try:
                    p.terminate()
                    p.wait(timeout=5) # Wait a bit for termination
                except subprocess.TimeoutExpired:
                    p.kill() # Force kill if it doesn't terminate
                    p.wait()
                except Exception as e:
                    print(f"Error terminating subprocess: {e}")
                yield all_videos.copy(), None, "Generation stopped by user.", ""
                return

            line = p.stdout.readline()
            if not line:
                if p.poll() is not None: break # Process finished
                time.sleep(0.01); continue # Wait for more output

            line = line.strip()
            if not line: continue
            print(f"SUBPROCESS: {line}") # Log subprocess output

# --- Check for Section Start Log ---
            # More robust regex that captures Section X/Y regardless of prefixes or suffixes
            section_match = re.search(r"---.*?Section\s+(\d+)\s*/\s*(\d+)(?:\s+|$|\()", line)
            tqdm_match = re.search(r'(\d+)\%\|.+\| (\d+)/(\d+) \[(\d{2}:\d{2})<(\d{2}:\d{2})', line)
            phase_changed = False
            if section_match:
                # Directly extract current section number and total sections
                current_section_num_display = int(section_match.group(1))
                total_sections_from_log = int(section_match.group(2))
                # Update state variables
                display_section_num = current_section_num_display
                # Update actual total only if it changes or hasn't been set
                if actual_total_sections != total_sections_from_log:
                    actual_total_sections = total_sections_from_log
                    print(f"Detected/Updated actual total sections: {actual_total_sections}")
                # Update phase and status/progress text
                new_phase = f"Generating Section {display_section_num}"
                if current_phase != new_phase:
                    current_phase = new_phase
                    phase_changed = True
                # Use the latest section info directly
                progress_text = f"Item {i+1}/{batch_size} | Section {display_section_num}/{actual_total_sections} | Preparing..."
                status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed}) - {current_phase}"
            # --- Process TQDM Progress ---
            elif tqdm_match:
                percentage = int(tqdm_match.group(1))
                current_step = int(tqdm_match.group(2))
                total_steps = int(tqdm_match.group(3))
                time_elapsed = tqdm_match.group(4)
                time_remaining = tqdm_match.group(5)
                # Use the last known section numbers. If no section log seen yet, use estimate.
                current_total_for_display = actual_total_sections if actual_total_sections is not None else total_sections_estimate
                section_str = f"Section {display_section_num}/{current_total_for_display}"
                # Update progress text with TQDM info and the current section string
                progress_text = f"Item {i+1}/{batch_size} | {section_str} | Step {current_step}/{total_steps} ({percentage}%) | Elapsed: {time_elapsed}, Remaining: {time_remaining}"
                denoising_phase = f"Denoising Section {display_section_num}"
                if current_phase != denoising_phase:
                    current_phase = denoising_phase
                    phase_changed = True
                status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed}) - {current_phase}"

            # --- Process Other Log Lines ---
            elif "Decoding video..." in line:
                 if current_phase != "Decoding Video":
                     current_phase = "Decoding Video"
                     phase_changed = True
                 progress_text = f"Item {i+1}/{batch_size} | {current_phase}..."
                 status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed}) - {current_phase}"

            elif "INFO:__main__:Video saved to:" in line:
                 match = re.search(r"Video saved to:\s*(.*\.mp4)", line)
                 if match:
                     found_video_path = match.group(1).strip()
                     if os.path.exists(found_video_path):
                         current_video_path = found_video_path
                         all_videos.append((current_video_path, f"Seed: {current_seed}"))
                         print(f"Video path found and added to gallery list: {current_video_path}")
                     else:
                          print(f"Warning: Parsed video path does not exist: {found_video_path}")
                     status_text = f"Video {i+1}/{batch_size} Saved (Seed: {current_seed})"
                     progress_text = f"Saved: {os.path.basename(found_video_path)}"
                     current_phase = "Saved"
                     phase_changed = True
                 else:
                     print(f"Warning: Could not parse video path from INFO line: {line}")

            elif "ERROR" in line.upper() or "TRACEBACK" in line.upper():
                 status_text = f"Item {i+1}/{batch_size}: Error Detected (Check Console)"
                 progress_text = line # Show the error line
                 if current_phase != "Error":
                    current_phase = "Error"
                    phase_changed = True

            # --- Update status_text if phase changed implicitly ---
            # This covers cases where phase changes but isn't explicitly set above (e.g., transitioning between tqdm steps within the same section)
            elif phase_changed and current_phase not in ["Saved", "Error"]:
                 status_text = f"Generating video {i + 1} of {batch_size} (Seed: {current_seed}) - {current_phase}"

            preview_updated = False
            current_mtime = 0
            found_preview_path = None

            if enable_preview:
                # Check MP4 first, then PNG
                if os.path.exists(preview_mp4_path):
                    current_mtime = os.path.getmtime(preview_mp4_path)
                    found_preview_path = preview_mp4_path
                elif os.path.exists(preview_png_path):
                    current_mtime = os.path.getmtime(preview_png_path)
                    found_preview_path = preview_png_path

                if found_preview_path and current_mtime > last_preview_mtime:
                    print(f"DEBUG: Preview file updated: {found_preview_path} (mtime: {current_mtime})")
                    # --- FIX: Yield the clean path ---
                    current_preview_yield_path = found_preview_path # REMOVED the cache buster
                    # --- END FIX ---
                    last_preview_mtime = current_mtime
                    preview_updated = True
            # --- End Preview Check ---

            # --- YIELD ---
            # Yield progress and potentially updated clean preview path
            yield all_videos.copy(), current_preview_yield_path, status_text, progress_text

        # --- Subprocess Finished ---
        p.stdout.close(); rc = p.wait()
        clear_cuda_cache(); time.sleep(0.1)

        # --- Collect Output and Save Metadata (If successful) ---
        if rc == 0 and current_video_path and os.path.exists(current_video_path):
            # Create metadata dictionary matching backend arguments
            parameters = {
                "prompt": prompt, "negative_prompt": negative_prompt,
                "input_image": os.path.basename(input_image) if input_image else None,
                 # --- ADDED: Metadata for Section Control ---
                "section_controls": [
                     {"index": s, "prompt_override": p, "image_override": os.path.basename(i) if i else None}
                     for s, p, i in zip(framepack_secs, framepack_sec_prompts, framepack_sec_images)
                     if (p and p.strip()) or i # Only record if prompt or image was set
                 ],
                "final_prompt_arg": final_prompt_arg, # Store the actual argument passed
                "final_image_path_arg": final_image_path_arg, # Store the actual argument passed
                "input_end_frame": os.path.basename(input_end_frame) if input_end_frame else None,
                # Model Paths
                "transformer_path": transformer_path,
                "vae_path": vae_path,
                "text_encoder_path": text_encoder_path,
                "text_encoder_2_path": text_encoder_2_path,
                "image_encoder_path": image_encoder_path,
                # Core Params
                "video_width": final_width, "video_height": final_height,
                "video_seconds": total_second_length, "fps": fps, "seed": current_seed,
                "infer_steps": steps, "embedded_cfg_scale": distilled_guidance_scale,
                "guidance_scale": cfg, "guidance_rescale": rs, "sample_solver": sample_solver,
                "latent_window_size": latent_window_size,
                 # --- Add End Frame Blending Params to Metadata ---
                #"end_frame_influence": end_frame_influence if input_end_frame else "none",
                #"end_frame_weight": end_frame_weight if input_end_frame else 0.0,
                # --- End Metadata Addition ---
                # Performance/Memory
                "fp8": fp8,
                "fp8_scaled": fp8_scaled,
                "fp8_llm": fp8_llm,
                "blocks_to_swap": blocks_to_swap,
                "bulk_decode": bulk_decode,
                "attn_mode": attn_mode,
                "vae_chunk_size": vae_chunk_size,
                "vae_spatial_tile_sample_min_size": vae_spatial_tile_sample_min_size,
                "device": device,
                # LoRA
                "lora_weights": [os.path.basename(p) for p in valid_loras_paths],
                "lora_multipliers": [float(m) for m in valid_loras_mults],
                "original_dims_str": original_dims_str,
                "target_resolution": target_resolution,
                "is_f1": is_f1
            }
            try:
                add_metadata_to_video(current_video_path, parameters)
                print(f"Added metadata to {current_video_path}")
            except Exception as meta_err:
                print(f"Warning: Failed to add metadata to {current_video_path}: {meta_err}")

            status_text = f"Completed (Seed: {current_seed})"
            progress_text = f"Video saved to: {os.path.basename(current_video_path)}"
            yield all_videos, current_preview_yield_path, status_text, progress_text
        elif rc != 0:
            status_text = f"Failed (Seed: {current_seed})"
            progress_text = f"Subprocess failed with exit code {rc}. Check console logs."
            yield all_videos.copy(), current_preview_yield_path, status_text, progress_text
            # Don't add to all_videos if failed
        else: # rc == 0 but current_video_path is not set or doesn't exist
            status_text = f"Failed (Seed: {current_seed})"
            progress_text = "Subprocess finished, but could not confirm generated video file. Check logs."
    yield all_videos, current_preview_yield_path, "FramePack Batch complete", ""

    time.sleep(0.2) # Small delay between batch items

    # --- Final Yield ---
    yield all_videos, current_preview_yield_path, "FramePack Batch complete", ""

def calculate_framepack_width(height, original_dims):
    """Calculate FramePack width based on height maintaining aspect ratio (divisible by 32)"""
    if not original_dims or height is None:
        return gr.update()
    try:
        # Ensure height is an integer and divisible by 32
        height = int(height)
        if height <= 0 : return gr.update()
        height = (height // 32) * 32 # <-- Use 32
        height = max(64, height) # Min height (64 is divisible by 32)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_h == 0: return gr.update()
        aspect_ratio = orig_w / orig_h
        # Calculate new width, rounding to the nearest multiple of 32
        new_width = round((height * aspect_ratio) / 32) * 32 # <-- Round and use 32
        return gr.update(value=max(64, new_width)) # Ensure minimum size (also divisible by 32)

    except Exception as e:
        print(f"Error calculating width: {e}")
        return gr.update()

def calculate_framepack_height(width, original_dims):
    """Calculate FramePack height based on width maintaining aspect ratio (divisible by 32)"""
    if not original_dims or width is None:
        return gr.update()
    try:
        # Ensure width is an integer and divisible by 32
        width = int(width)
        if width <= 0: return gr.update()
        width = (width // 32) * 32 # <-- Use 32
        width = max(64, width) # Min width (64 is divisible by 32)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_w == 0: return gr.update()
        aspect_ratio = orig_w / orig_h
        # Calculate new height, rounding to the nearest multiple of 32
        new_height = round((width / aspect_ratio) / 32) * 32 # <-- Round and use 32
        return gr.update(value=max(64, new_height)) # Ensure minimum size (also divisible by 32)
    except Exception as e:
        print(f"Error calculating height: {e}")
        return gr.update()

def update_framepack_from_scale(scale, original_dims):
    """Update FramePack dimensions based on scale percentage (divisible by 32)"""
    if not original_dims:
        return gr.update(), gr.update(), gr.update()
    try:
        scale = float(scale) if scale is not None else 100.0
        if scale <= 0: scale = 100.0

        orig_w, orig_h = map(int, original_dims.split('x'))
        scale_factor = scale / 100.0

        # Calculate and round to the nearest multiple of 32
        new_w = round((orig_w * scale_factor) / 32) * 32 # <-- Round and use 32
        new_h = round((orig_h * scale_factor) / 32) * 32 # <-- Round and use 32

        # Ensure minimum size (must be multiple of 32)
        new_w = max(64, new_w) # 64 is divisible by 32
        new_h = max(64, new_h)

        # Clear target resolution if using scale slider for explicit dims
        return gr.update(value=new_w), gr.update(value=new_h), gr.update(value=None)
    except Exception as e:
        print(f"Error updating from scale: {e}")
        return gr.update(), gr.update(), gr.update()

def process_i2v_single_video(
    prompt: str,
    image_path: str,
    width: int,
    height: int,
    batch_size: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    dit_folder: str,
    model: str,
    vae: str,
    te1: str,
    te2: str,
    clip_vision_path: str,
    save_path: str,
    flow_shift: float,
    cfg_scale: float, # embedded_cfg_scale
    guidance_scale: float, # main CFG
    output_type: str,
    attn_mode: str,
    block_swap: int,
    exclude_single_blocks: bool,
    use_split_attn: bool,
    lora_folder: str,
    vae_chunk_size: int,
    vae_spatial_tile_min: int,
    # --- Explicit LoRA args instead of *lora_params ---
    lora1: str = "None",
    lora2: str = "None",
    lora3: str = "None",
    lora4: str = "None",
    lora1_multiplier: float = 1.0,
    lora2_multiplier: float = 1.0,
    lora3_multiplier: float = 1.0,
    lora4_multiplier: float = 1.0,
    # --- End LoRA args ---
    negative_prompt: Optional[str] = None,
    use_fp8: bool = False,
    fp8_llm: bool = False
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate a single video using hv_i2v_generate_video.py"""
    global stop_event

    # ... (Keep existing argument validation and env setup) ...
    if stop_event.is_set():
        yield [], "", ""
        return

    # Argument validation
    if not image_path or not os.path.exists(image_path):
         yield [], "Error: Input image not found", f"Cannot find image: {image_path}"
         return
    # Check clip vision path only if needed (Hunyuan-I2V, not SkyReels-I2V based on script name)
    is_hunyuan_i2v = "mp_rank_00_model_states_i2v" in model # Heuristic check
    if is_hunyuan_i2v and (not clip_vision_path or not os.path.exists(clip_vision_path)):
         yield [], "Error: CLIP Vision model not found", f"Cannot find file: {clip_vision_path}"
         return

    if os.path.isabs(model):
        model_path = model
    else:
        model_path = os.path.normpath(os.path.join(dit_folder, model))

    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)
    else:
        current_seed = seed

    clear_cuda_cache()

    command = [
        sys.executable,
        "hv_i2v_generate_video.py", # <<< Use the new script
        "--dit", model_path,
        "--vae", vae,
        "--text_encoder1", te1,
        "--text_encoder2", te2,
        # Add clip vision path only if it's likely the Hunyuan I2V model
        *(["--clip_vision_path", clip_vision_path] if is_hunyuan_i2v else []),
        "--prompt", prompt,
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", save_path,
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--embedded_cfg_scale", str(cfg_scale),
        "--guidance_scale", str(guidance_scale),
        "--output_type", output_type,
        "--attn_mode", attn_mode,
        "--blocks_to_swap", str(block_swap),
        "--image_path", image_path
    ]

    if negative_prompt:
        command.extend(["--negative_prompt", negative_prompt])

    if use_fp8:
        command.append("--fp8")
    if fp8_llm:
        command.append("--fp8_llm")

    if exclude_single_blocks:
        command.append("--exclude_single_blocks")
    if use_split_attn:
        command.append("--split_attn")

    if vae_chunk_size > 0:
        command.extend(["--vae_chunk_size", str(vae_chunk_size)])
    if vae_spatial_tile_min > 0:
        command.extend(["--vae_spatial_tile_sample_min_size", str(vae_spatial_tile_min)])

    # --- Updated LoRA handling using named arguments ---
    lora_weights_list = [lora1, lora2, lora3, lora4]
    lora_multipliers_list = [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]
    valid_loras = []
    for weight, mult in zip(lora_weights_list, lora_multipliers_list):
        if weight and weight != "None":
            lora_file_path = os.path.join(lora_folder, weight)
            if os.path.exists(lora_file_path):
                 valid_loras.append((lora_file_path, mult))
            else:
                print(f"Warning: LoRA file not found: {lora_file_path}")

    if valid_loras:
        weights = [weight for weight, _ in valid_loras]
        multipliers = [str(mult) for _, mult in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)
    # --- End Updated LoRA handling ---

    # ... (Keep subprocess execution, output collection, and metadata saving logic) ...
    command_str = [str(c) for c in command] # Ensure all args are strings
    print(f"Running Command (I2V): {' '.join(command_str)}")

    p = subprocess.Popen(
        command_str, # Use stringified command
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = []

    while True:
        if stop_event.is_set():
            p.terminate()
            p.wait()
            yield videos, current_previews, "Generation stopped by user.", ""
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue

        print(line, end='') # Print progress to console
        if '|' in line and '%' in line and '[' in line and ']' in line:
            yield videos.copy(), f"Processing (seed: {current_seed})", line.strip()

    p.stdout.close()
    p.wait()

    clear_cuda_cache()
    time.sleep(0.5)

    # Collect generated video
    save_path_abs = os.path.abspath(save_path)
    generated_video_path = None
    if os.path.exists(save_path_abs):
        all_videos_files = sorted(
            [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
            key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
            reverse=True
        )
        # Try to find the video matching the seed
        matching_videos = [v for v in all_videos_files if f"_{current_seed}" in v]
        if matching_videos:
            generated_video_path = os.path.join(save_path_abs, matching_videos[0])

    if generated_video_path:
         # Collect parameters for metadata (adjust as needed for i2v specifics)
        parameters = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "video_length": video_length,
            "fps": fps,
            "infer_steps": infer_steps,
            "seed": current_seed,
            "model": model,
            "vae": vae,
            "te1": te1,
            "te2": te2,
            "clip_vision_path": clip_vision_path,
            "save_path": save_path,
            "flow_shift": flow_shift,
            "embedded_cfg_scale": cfg_scale,
            "guidance_scale": guidance_scale,
            "output_type": output_type,
            "attn_mode": attn_mode,
            "block_swap": block_swap,
            "lora_weights": list(lora_weights_list), # Save the list
            "lora_multipliers": list(lora_multipliers_list), # Save the list
            "input_image": image_path,
            "negative_prompt": negative_prompt if negative_prompt else None,
            "vae_chunk_size": vae_chunk_size,
            "vae_spatial_tile_min": vae_spatial_tile_min,
            "use_fp8_dit": use_fp8,
            "use_fp8_llm": fp8_llm
        }
        add_metadata_to_video(generated_video_path, parameters)
        videos.append((str(generated_video_path), f"Seed: {current_seed}"))
        yield videos, f"Completed (seed: {current_seed})", ""
    else:
        yield [], f"Failed (seed: {current_seed})", "Could not find generated video file."


def process_i2v_batch(
    prompt: str,
    image_path: str,
    width: int,
    height: int,
    batch_size: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    dit_folder: str,
    model: str,
    vae: str,
    te1: str,
    te2: str,
    clip_vision_path: str, # Added
    save_path: str,
    flow_shift: float,
    cfg_scale: float, # embedded_cfg_scale
    guidance_scale: float, # main CFG
    output_type: str,
    attn_mode: str,
    block_swap: int,
    exclude_single_blocks: bool,
    use_split_attn: bool,
    lora_folder: str,
    vae_chunk_size: int, # Added
    vae_spatial_tile_min: int, # Added
    negative_prompt: Optional[str] = None, # Added
    use_fp8: bool = False, # Added
    fp8_llm: bool = False, # Added
    *lora_params # Captures LoRA weights and multipliers
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Process a batch of videos using the new I2V script"""
    global stop_event
    stop_event.clear()

    all_videos = []
    progress_text = "Starting I2V generation..."
    yield [], "Preparing...", progress_text

    # Extract LoRA weights and multipliers once
    num_lora_weights = 4
    lora_weights_list = lora_params[:num_lora_weights]
    lora_multipliers_list = lora_params[num_lora_weights:num_lora_weights*2]

    for i in range(batch_size):
        if stop_event.is_set():
            yield all_videos, "Generation stopped by user.", ""
            return

        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1:
            current_seed = seed + i

        batch_text = f"Generating video {i + 1} of {batch_size} (I2V)"
        yield all_videos.copy(), batch_text, progress_text

        # Call the single video processing function
        single_gen = process_i2v_single_video(
            prompt=prompt,
            image_path=image_path,
            width=width,
            height=height,
            batch_size=batch_size,
            video_length=video_length,
            fps=fps,
            infer_steps=infer_steps,
            seed=current_seed,
            dit_folder=dit_folder,
            model=model,
            vae=vae,
            te1=te1,
            te2=te2,
            clip_vision_path=clip_vision_path,
            save_path=save_path,
            flow_shift=flow_shift,
            cfg_scale=cfg_scale,
            guidance_scale=guidance_scale,
            output_type=output_type,
            attn_mode=attn_mode,
            block_swap=block_swap,
            exclude_single_blocks=exclude_single_blocks,
            use_split_attn=use_split_attn,
            lora_folder=lora_folder,
            vae_chunk_size=vae_chunk_size,
            vae_spatial_tile_min=vae_spatial_tile_min,
            # --- Pass LoRA params by keyword ---
            lora1=lora_weights_list[0],
            lora2=lora_weights_list[1],
            lora3=lora_weights_list[2],
            lora4=lora_weights_list[3],
            lora1_multiplier=lora_multipliers_list[0],
            lora2_multiplier=lora_multipliers_list[1],
            lora3_multiplier=lora_multipliers_list[2],
            lora4_multiplier=lora_multipliers_list[3],
            # --- End LoRA keyword args ---
            negative_prompt=negative_prompt,
            use_fp8=use_fp8,
            fp8_llm=fp8_llm
        )

        # Yield progress updates from the single generator
        try:
            for videos, status, progress in single_gen:
                if videos:
                    # Only add the latest video from this specific generation
                    new_video = videos[-1]
                    if new_video not in all_videos:
                         all_videos.append(new_video)
                yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress
        except Exception as e:
             yield all_videos.copy(), f"Error in batch {i+1}: {e}", ""
             print(f"Error during single I2V generation: {e}") # Log error

        # Optional small delay between batch items
        time.sleep(0.1)

    yield all_videos, "I2V Batch complete", ""


def wanx_extend_video_wrapper(
    prompt, negative_prompt, input_image, base_video_path,
    width, height, video_length, fps, infer_steps,
    flow_shift, guidance_scale, seed,
    task, dit_folder, dit_path, vae_path, t5_path, clip_path, # <--- Parameters received here
    save_path, output_type, sample_solver, exclude_single_blocks,
    attn_mode, block_swap, fp8, fp8_scaled, fp8_t5, lora_folder,
    slg_layers="", slg_start=0.0, slg_end=1.0,
    lora1="None", lora2="None", lora3="None", lora4="None",
    lora1_multiplier=1.0, lora2_multiplier=1.0, lora3_multiplier=1.0, lora4_multiplier=1.0,
    enable_cfg_skip=False, cfg_skip_mode="none", cfg_apply_ratio=0.7
):
    """Direct wrapper that bypasses the problematic wanx_generate_video function"""
    global stop_event

    # All videos generated
    all_videos = []

    # Debug prints to understand what we're getting
    print(f"DEBUG - Received parameters in wanx_extend_video_wrapper:")
    print(f"  task: {task}")
    print(f"  dit_folder: {dit_folder}") # <<< Should be the folder path ('wan')
    print(f"  dit_path: {dit_path}")     # <<< Should be the model filename
    print(f"  vae_path: {vae_path}")     # <<< Should be the VAE path
    print(f"  t5_path: {t5_path}")       # <<< Should be the T5 path
    print(f"  clip_path: {clip_path}")     # <<< Should be the CLIP path
    print(f"  output_type: {output_type}")
    print(f"  sample_solver: {sample_solver}")
    print(f"  attn_mode: {attn_mode}")
    print(f"  block_swap: {block_swap}")

    # Get current seed
    current_seed = seed
    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)

    # --- START CRITICAL FIX ---
    # Detect if parameters are swapped based on the pattern observed in the error log
    # Check if dit_path looks like a VAE path (contains "VAE" or ends with .pth)
    # AND dit_folder looks like a model filename (ends with .safetensors or .pt)
    params_swapped = False
    if dit_path and dit_folder and \
       (("VAE" in dit_path or dit_path.endswith(".pth")) and \
        (dit_folder.endswith(".safetensors") or dit_folder.endswith(".pt"))):
        params_swapped = True
        print("WARNING: Parameters appear to be swapped in extend workflow. Applying correction...")

        # Correct the parameters based on the observed swap
        actual_model_filename = dit_folder # Original dit_folder was the filename
        actual_vae_path = dit_path         # Original dit_path was the VAE path
        actual_t5_path = vae_path          # Original vae_path was the T5 path
        actual_clip_path = t5_path         # Original t5_path was the CLIP path

        # Assign corrected values back to expected variable names for the rest of the function
        dit_path = actual_model_filename
        vae_path = actual_vae_path
        t5_path = actual_t5_path
        clip_path = actual_clip_path
        dit_folder = "wan" # Assume default 'wan' folder if swapped

        print(f"  Corrected dit_folder: {dit_folder}")
        print(f"  Corrected dit_path (model filename): {dit_path}")
        print(f"  Corrected vae_path: {vae_path}")
        print(f"  Corrected t5_path: {t5_path}")
        print(f"  Corrected clip_path: {clip_path}")

    # Construct the full model path using the potentially corrected dit_folder and dit_path
    actual_model_path = os.path.join(dit_folder, dit_path) if not os.path.isabs(dit_path) else dit_path
    print(f"  Using actual_model_path for --dit: {actual_model_path}")
    # --- END CRITICAL FIX ---

    # Prepare environment
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    # Clear CUDA cache
    clear_cuda_cache()

    # Validate and fix parameters
    # Fix output_type - must be one of: video, images, latent, both
    valid_output_types = ["video", "images", "latent", "both"]
    actual_output_type = "video" if output_type not in valid_output_types else output_type

    # Fix sample_solver - must be one of: unipc, dpm++, vanilla
    valid_sample_solvers = ["unipc", "dpm++", "vanilla"]
    actual_sample_solver = "unipc" if sample_solver not in valid_sample_solvers else sample_solver

    # Fix attn_mode - must be one of: sdpa, flash, sageattn, xformers, torch
    valid_attn_modes = ["sdpa", "flash", "sageattn", "xformers", "torch"]
    actual_attn_mode = "sdpa" if attn_mode not in valid_attn_modes else attn_mode

    # Fix block_swap - must be an integer
    try:
        actual_block_swap = int(block_swap)
    except (ValueError, TypeError):
        actual_block_swap = 0

    # Build command array with explicit string conversions for EVERY parameter
    command = [
        sys.executable,
        "wan_generate_video.py",
        "--task", str(task),
        "--prompt", str(prompt),
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", str(save_path),
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--guidance_scale", str(guidance_scale),
        "--output_type", actual_output_type,
        "--sample_solver", actual_sample_solver,
        "--attn_mode", actual_attn_mode,
        "--blocks_to_swap", str(actual_block_swap),
        # Use the corrected model path and other paths
        "--dit", str(actual_model_path), # <<< Use corrected full model path
        "--vae", str(vae_path),          # <<< Use potentially corrected vae_path
        "--t5", str(t5_path)            # <<< Use potentially corrected t5_path
    ]

    # Add image path and clip model path if needed
    if input_image:
        command.extend(["--image_path", str(input_image)])
        # Use the potentially corrected clip_path
        if clip_path and clip_path != "outputs" and "output" not in clip_path:
            command.extend(["--clip", str(clip_path)]) # <<< Use potentially corrected clip_path

    # Add negative prompt
    if negative_prompt:
        command.extend(["--negative_prompt", str(negative_prompt)])

    # Handle boolean flags - keep original values
    if fp8:
        command.append("--fp8")

    if fp8_scaled:
        command.append("--fp8_scaled")

    if fp8_t5:
        command.append("--fp8_t5")

    # Add SLG parameters
    try:
        # Ensure slg_layers is treated as a string before splitting
        slg_layers_str = str(slg_layers) if slg_layers is not None else ""
        if slg_layers_str and slg_layers_str.strip() and slg_layers_str.lower() != "none":
            slg_list = []
            for layer in slg_layers_str.split(","):
                layer = layer.strip()
                if layer.isdigit():  # Only add if it's a valid integer
                    slg_list.append(int(layer))
            if slg_list:  # Only add if we have valid layers
                command.extend(["--slg_layers", ",".join(map(str, slg_list))])

                # Only add slg_start and slg_end if we have valid slg_layers
                if slg_start is not None:
                    try:
                         slg_start_float = float(slg_start)
                         if slg_start_float >= 0:
                             command.extend(["--slg_start", str(slg_start_float)])
                    except (ValueError, TypeError): pass # Ignore if conversion fails
                if slg_end is not None:
                     try:
                         slg_end_float = float(slg_end)
                         if slg_end_float <= 1.0:
                             command.extend(["--slg_end", str(slg_end_float)])
                     except (ValueError, TypeError): pass # Ignore if conversion fails
    except Exception as e: # Catch potential errors during processing
        print(f"Warning: Error processing SLG parameters: {e}")
        pass

    # Handle LoRA weights and multipliers
    valid_loras = []
    if lora_folder and isinstance(lora_folder, str):
        for weight, mult in zip([lora1, lora2, lora3, lora4],
                              [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]):
            # Skip None or empty values
            if not weight or str(weight).lower() == "none":
                continue

            # Construct path and check existence
            full_path = os.path.join(str(lora_folder), str(weight))
            if not os.path.exists(full_path):
                print(f"LoRA file not found: {full_path}")
                continue

            # Add valid LoRA
            valid_loras.append((full_path, str(mult)))

    if valid_loras:
        weights = [w for w, _ in valid_loras]
        multipliers = [m for _, m in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)

    # Final conversion to ensure all elements are strings
    command_str = [str(item) for item in command]

    print(f"Running Command (wanx_extend_video_wrapper): {' '.join(command_str)}")

    # Process execution
    p = subprocess.Popen(
        command_str, # Use stringified command
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = [] # Store the generated (non-extended) video first

    # Process stdout in real time
    while True:
        if stop_event.is_set():
            p.terminate()
            p.wait()
            yield [], "", "Generation stopped by user."
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue

        print(line, end='')
        if '|' in line and '%' in line and '[' in line and ']' in line:
             # Yield empty list during processing, actual video is collected later
            yield [], f"Processing (seed: {current_seed})", line.strip()

    p.stdout.close()
    return_code = p.wait() # Get return code

    # Clean CUDA cache and wait
    clear_cuda_cache()
    time.sleep(0.5)

    # Check return code
    if return_code != 0:
        print(f"❌ Error: wan_generate_video.py exited with code {return_code}")
        yield [], f"Failed (seed: {current_seed})", f"Subprocess failed with code {return_code}"
        return

    # Find the *newly generated* video first
    generated_video_path = None
    save_path_abs = os.path.abspath(save_path)
    if os.path.exists(save_path_abs):
        # Find the most recent mp4 containing the seed
        all_mp4_files = glob.glob(os.path.join(save_path_abs, f"*_{current_seed}*.mp4"))
        if all_mp4_files:
            generated_video_path = max(all_mp4_files, key=os.path.getmtime)
            print(f"Found newly generated video: {generated_video_path}")

            # Add metadata to the generated video before potential concatenation
            parameters = {
                "prompt": prompt, "negative_prompt": negative_prompt, "input_image": input_image,
                "width": width, "height": height, "video_length": video_length, "fps": fps,
                "infer_steps": infer_steps, "flow_shift": flow_shift, "guidance_scale": guidance_scale,
                "seed": current_seed, "task": task, "dit_path": actual_model_path, # Store the actual path used
                "vae_path": vae_path, "t5_path": t5_path, "clip_path": clip_path,
                "save_path": save_path, "output_type": actual_output_type, "sample_solver": actual_sample_solver,
                "exclude_single_blocks": exclude_single_blocks, "attn_mode": actual_attn_mode,
                "block_swap": actual_block_swap, "fp8": fp8, "fp8_scaled": fp8_scaled, "fp8_t5": fp8_t5,
                "lora_weights": [lora1, lora2, lora3, lora4],
                "lora_multipliers": [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier],
                "slg_layers": slg_layers, "slg_start": slg_start, "slg_end": slg_end,
                "is_extension_source": True # Flag this as the source for an extension
            }
            add_metadata_to_video(generated_video_path, parameters)
            # videos.append((str(generated_video_path), f"Generated segment (Seed: {current_seed})")) # Optionally yield segment
        else:
             print(f"Could not find generated video segment for seed {current_seed} in {save_path_abs}")

    # Stop here if no new video segment was generated
    if not generated_video_path:
        yield [], f"Failed (seed: {current_seed})", "Could not find generated video segment."
        return

    # Now concatenate with base video if we have the new segment and a base_video_path
    if generated_video_path and base_video_path and os.path.exists(base_video_path):
        try:
            print(f"Extending base video: {base_video_path}")

            # Create unique output filename for the *extended* video
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
            output_filename = f"extended_{timestamp}_seed{current_seed}_{Path(base_video_path).stem}.mp4"
            output_path = os.path.join(save_path_abs, output_filename)

            # Create a temporary file list for ffmpeg concatenation
            list_file = os.path.join(save_path_abs, f"temp_concat_list_{current_seed}.txt")
            with open(list_file, "w") as f:
                f.write(f"file '{os.path.abspath(base_video_path)}'\n")
                f.write(f"file '{os.path.abspath(generated_video_path)}'\n") # Use the newly generated segment

            print(f"Concatenating: {base_video_path} + {generated_video_path} -> {output_path}")

            # Run ffmpeg concatenation command
            concat_command = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",       # Allow relative paths if needed, but we use absolute
                "-i", list_file,
                "-c", "copy",       # Fast concatenation without re-encoding
                "-y",               # Overwrite output if exists
                output_path
            ]

            # Convert all command parts to strings
            concat_command_str = [str(item) for item in concat_command]

            print(f"Running FFmpeg command: {' '.join(concat_command_str)}")
            concat_result = subprocess.run(concat_command_str, check=False, capture_output=True, text=True) # Don't check=True initially

            # Clean up temporary list file
            if os.path.exists(list_file):
                try:
                    os.remove(list_file)
                except OSError as e:
                    print(f"Warning: Could not remove temp list file {list_file}: {e}")


            # Check if concatenation was successful
            if concat_result.returncode == 0 and os.path.exists(output_path):
                # Optionally, add metadata to the *extended* video as well
                extended_parameters = parameters.copy()
                extended_parameters["is_extension_source"] = False
                extended_parameters["base_video"] = os.path.basename(base_video_path)
                add_metadata_to_video(output_path, extended_parameters)

                extended_video_gallery_item = [(output_path, f"Extended (Seed: {current_seed})")]
                print(f"✅ Successfully created extended video: {output_path}")
                yield extended_video_gallery_item, "Extended video created successfully", ""
                return # Success!
            else:
                print(f"❌ Failed to create extended video at {output_path}")
                print(f"FFmpeg stderr: {concat_result.stderr}")
                # Yield the generated segment if concatenation failed
                yield [(generated_video_path, f"Generated segment (Seed: {current_seed})")], "Generated segment (extension failed)", f"FFmpeg failed: {concat_result.stderr[:200]}..."
                return

        except Exception as e:
            print(f"❌ Error during concatenation: {str(e)}")
            # Yield the generated segment if concatenation failed
            yield [(generated_video_path, f"Generated segment (Seed: {current_seed})")], "Generated segment (extension error)", f"Error: {str(e)}"
            return

    # If we got here, base_video_path was likely None or didn't exist, but generation succeeded
    yield [(generated_video_path, f"Generated segment (Seed: {current_seed})")], "Generated segment (no base video provided)", ""

def wanx_v2v_generate_video(
    prompt, 
    negative_prompt,
    input_video,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    strength,
    seed,
    task,
    dit_folder,
    dit_path,
    vae_path,
    t5_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers,
    slg_start,
    slg_end,
    lora1="None",
    lora2="None",
    lora3="None",
    lora4="None",
    lora1_multiplier=1.0,
    lora2_multiplier=1.0,
    lora3_multiplier=1.0,
    lora4_multiplier=1.0,
    enable_cfg_skip=False,
    cfg_skip_mode="none",
    cfg_apply_ratio=0.7,
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate video with WanX model in video-to-video mode"""
    global stop_event
    
    # Convert values safely to float or None
    try:
        slg_start_float = float(slg_start) if slg_start is not None and str(slg_start).lower() != "none" else None
    except (ValueError, TypeError):
        slg_start_float = None
        print(f"Warning: Could not convert slg_start '{slg_start}' to float")
    
    try:
        slg_end_float = float(slg_end) if slg_end is not None and str(slg_end).lower() != "none" else None
    except (ValueError, TypeError):
        slg_end_float = None
        print(f"Warning: Could not convert slg_end '{slg_end}' to float")
    
    print(f"slg_start_float: {slg_start_float}, slg_end_float: {slg_end_float}")
    
    if stop_event.is_set():
        yield [], "", ""
        return

    # Check if we need input video (required for v2v)
    if not input_video:
        yield [], "Error: No input video provided", "Please provide an input video for video-to-video generation"
        return

    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)
    else:
        current_seed = seed

    # Prepare environment
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    
    clear_cuda_cache()

    # Construct full dit_path including folder - this is the fix
    full_dit_path = os.path.join(dit_folder, dit_path) if not os.path.isabs(dit_path) else dit_path

    command = [
        sys.executable,
        "wan_generate_video.py",
        "--task", task,
        "--prompt", prompt,
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", save_path,
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--guidance_scale", str(guidance_scale),
        "--output_type", output_type,
        "--attn_mode", attn_mode,
        "--blocks_to_swap", str(block_swap),
        "--dit", full_dit_path,  # Use full_dit_path instead of dit_path
        "--vae", vae_path,
        "--t5", t5_path,
        "--sample_solver", sample_solver,
        "--video_path", input_video,  # This is the key for v2v mode
        "--strength", str(strength)   # Strength parameter for v2v
    ]
    if enable_cfg_skip and cfg_skip_mode != "none":
        command.extend([
            "--cfg_skip_mode", cfg_skip_mode,
            "--cfg_apply_ratio", str(cfg_apply_ratio)
        ])
    # Handle SLG parameters
    if slg_layers and str(slg_layers).strip() and slg_layers.lower() != "none":
        try:
            # Parse SLG layers
            layer_list = [int(x) for x in str(slg_layers).split(",")]
            if layer_list:  # Only proceed if we have valid layer values
                command.extend(["--slg_layers", ",".join(map(str, layer_list))])
                
                # Only add slg_start and slg_end if we have valid slg_layers
                try:
                    if slg_start_float is not None and slg_start_float >= 0:
                        command.extend(["--slg_start", str(slg_start_float)])
                    if slg_end_float is not None and slg_end_float <= 1.0:
                        command.extend(["--slg_end", str(slg_end_float)])
                except ValueError as e:
                    print(f"Invalid SLG timing values: {str(e)}")
        except ValueError as e:
            print(f"Invalid SLG layers format: {slg_layers} - {str(e)}")
    
    if negative_prompt:
        command.extend(["--negative_prompt", negative_prompt])
    
    if fp8:
        command.append("--fp8")
    
    if fp8_scaled:
        command.append("--fp8_scaled")
    
    if fp8_t5:
        command.append("--fp8_t5")
        
    if exclude_single_blocks:
        command.append("--exclude_single_blocks")
    
    # Handle LoRA weights and multipliers
    lora_weights = [lora1, lora2, lora3, lora4]
    lora_multipliers = [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]
    
    valid_loras = []
    for weight, mult in zip(lora_weights, lora_multipliers):
        if weight and weight != "None":
            full_path = os.path.join(lora_folder, weight)
            if not os.path.exists(full_path):
                print(f"LoRA file not found: {full_path}")
                continue
            valid_loras.append((full_path, mult))

    if valid_loras:
        weights = [w for w, _ in valid_loras]
        multipliers = [str(m) for _, m in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)
    
    print(f"Running: {' '.join(command)}")

    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = []
    
    while True:
        if stop_event.is_set():
            p.terminate()
            p.wait()
            yield [], "", "Generation stopped by user."
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue
            
        print(line, end='')
        if '|' in line and '%' in line and '[' in line and ']' in line:
            yield videos.copy(), f"Processing (seed: {current_seed})", line.strip()

    p.stdout.close()
    p.wait()

    clear_cuda_cache()
    time.sleep(0.5)

    # Collect generated video
    save_path_abs = os.path.abspath(save_path)
    if os.path.exists(save_path_abs):
        all_videos = sorted(
            [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
            key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
            reverse=True
        )
        matching_videos = [v for v in all_videos if f"_{current_seed}" in v]
        if matching_videos:
            video_path = os.path.join(save_path_abs, matching_videos[0])
            
            # Collect parameters for metadata
            parameters = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "video_length": video_length,
                "fps": fps,
                "infer_steps": infer_steps,
                "seed": current_seed,
                "task": task,
                "flow_shift": flow_shift,
                "guidance_scale": guidance_scale,
                "output_type": output_type,
                "attn_mode": attn_mode,
                "block_swap": block_swap,
                "input_video": input_video,
                "strength": strength,
                "lora_weights": [lora1, lora2, lora3, lora4],
                "lora_multipliers": [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier],
                "dit_path": full_dit_path,  # Store the full path in metadata
                "vae_path": vae_path,
                "t5_path": t5_path,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "sample_solver": sample_solver
            }
            
            add_metadata_to_video(video_path, parameters)
            videos.append((str(video_path), f"Seed: {current_seed}"))

    yield videos, f"Completed (seed: {current_seed})", ""

def wanx_v2v_batch_handler(
    prompt,
    negative_prompt,
    input_video,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    strength,
    seed,
    batch_size,
    task,
    dit_folder,  # folder path
    dit_path,    # model filename
    vae_path,
    t5_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers: str,
    slg_start: Optional[str],
    slg_end: Optional[str],
    enable_cfg_skip: bool,
    cfg_skip_mode: str,
    cfg_apply_ratio: float,
    *lora_params
):
    """Handle batch generation for WanX v2v"""
    global stop_event
    stop_event.clear()
    
    # Extract LoRA parameters
    num_lora_weights = 4
    lora_weights = lora_params[:num_lora_weights]
    lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
    
    all_videos = []
    progress_text = "Starting generation..."
    yield [], "Preparing...", progress_text
    
    # Process each item in the batch
    for i in range(batch_size):
        if stop_event.is_set():
            yield all_videos, "Generation stopped by user", ""
            return
            
        # Calculate seed for this batch item
        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1:
            current_seed = seed + i
            
        batch_text = f"Generating video {i + 1} of {batch_size}"
        yield all_videos.copy(), batch_text, progress_text
        
        # Generate a single video
        for videos, status, progress in wanx_v2v_generate_video(
            prompt, 
            negative_prompt, 
            input_video, 
            width, 
            height, 
            video_length, 
            fps, 
            infer_steps, 
            flow_shift, 
            guidance_scale,
            strength,
            current_seed,
            task, 
            dit_folder,  # Pass folder path
            dit_path,    # Pass model filename
            vae_path, 
            t5_path, 
            save_path, 
            output_type, 
            sample_solver, 
            exclude_single_blocks,
            attn_mode, 
            block_swap, 
            fp8, 
            fp8_scaled, 
            fp8_t5,
            lora_folder,
            slg_layers,
            slg_start,
            slg_end,
            *lora_weights,
            *lora_multipliers,
            enable_cfg_skip,
            cfg_skip_mode,
            cfg_apply_ratio,
        ):
            if videos:
                all_videos.extend(videos)
            yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress
        
        # Clear CUDA cache between generations
        clear_cuda_cache()
        time.sleep(0.5)
    
    yield all_videos, "Batch complete", ""

def update_wanx_v2v_dimensions(video):
    """Update dimensions from uploaded video"""
    if video is None:
        return "", gr.update(value=832), gr.update(value=480)
    
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return "Error opening video", gr.update(), gr.update()
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Make dimensions divisible by 32
    w = (w // 32) * 32
    h = (h // 32) * 32
    
    return f"{w}x{h}", w, h

def send_wanx_v2v_to_hunyuan_v2v(
    gallery: list,
    prompt: str,
    selected_index: int,
    width: int,
    height: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    flow_shift: float,
    guidance_scale: float,
    negative_prompt: str
) -> Tuple:
    """Send the selected WanX v2v video to Hunyuan v2v tab"""
    if gallery is None or not gallery:
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)
    
    # If no selection made but we have videos, use the first one
    if selected_index is None and len(gallery) > 0:
        selected_index = 0
        
    if selected_index is None or selected_index >= len(gallery):
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)

    selected_item = gallery[selected_index]
    
    # Handle different gallery item formats
    if isinstance(selected_item, tuple):
        video_path = selected_item[0]
    elif isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    else:
        video_path = selected_item

    # Clean up path for Video component
    if isinstance(video_path, tuple):
        video_path = video_path[0]
        
    # Make sure it's a string
    video_path = str(video_path)

    return (video_path, prompt, width, height, video_length, fps, infer_steps, seed, 
            flow_shift, guidance_scale, negative_prompt)

def handle_wanx_v2v_gallery_select(evt: gr.SelectData) -> int:
    """Track selected index when gallery item is clicked"""
    return evt.index

def variance_of_laplacian(image):
    """
    Compute the variance of the Laplacian of the image.
    Higher variance indicates a sharper image.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_sharpest_frame(video_path, frames_to_check=30):
    """
    Extract the sharpest frame from the last N frames of the video.
    
    Args:
        video_path (str): Path to the video file
        frames_to_check (int): Number of frames from the end to check
        
    Returns:
        tuple: (temp_image_path, frame_number, sharpness_score)
    """
    print(f"\n=== Extracting sharpest frame from the last {frames_to_check} frames ===")
    print(f"Input video path: {video_path}")
    
    if not video_path or not os.path.exists(video_path):
        print("❌ Error: Video file does not exist")
        return None, None, None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Failed to open video file")
            return None, None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Total frames detected: {total_frames}, FPS: {fps:.2f}")
        
        if total_frames < 1:
            print("❌ Error: Video contains 0 frames")
            return None, None, None
        
        # Determine how many frames to check (the last N frames)
        if frames_to_check > total_frames:
            frames_to_check = total_frames
            start_frame = 0
        else:
            start_frame = total_frames - frames_to_check
        
        print(f"Checking frames {start_frame} to {total_frames-1}")
        
        # Find the sharpest frame
        sharpest_frame = None
        max_sharpness = -1
        sharpest_frame_number = -1
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames with a progress bar
        with tqdm(total=frames_to_check, desc="Finding sharpest frame") as pbar:
            frame_idx = start_frame
            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and calculate sharpness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = variance_of_laplacian(gray)
                
                # Update if this is the sharpest frame so far
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    sharpest_frame = frame.copy()
                    sharpest_frame_number = frame_idx
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        if sharpest_frame is None:
            print("❌ Error: Failed to find a sharp frame")
            return None, None, None
        
        # Prepare output path
        temp_dir = os.path.abspath("temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"sharpest_frame_{os.path.basename(video_path)}.png")
        print(f"Saving frame to: {temp_path}")
        
        # Write and verify
        if not cv2.imwrite(temp_path, sharpest_frame):
            print("❌ Error: Failed to write frame to file")
            return None, None, None
            
        if not os.path.exists(temp_path):
            print("❌ Error: Output file not created")
            return None, None, None
        
        # Calculate frame time in seconds
        frame_time = sharpest_frame_number / fps
        
        print(f"✅ Extracted sharpest frame: {sharpest_frame_number} (at {frame_time:.2f}s) with sharpness {max_sharpness:.2f}")
        return temp_path, sharpest_frame_number, max_sharpness

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None, None, None
    finally:
        if 'cap' in locals():
            cap.release()

def trim_video_to_frame(video_path, frame_number, output_dir="outputs"):
    """
    Trim video up to the specified frame and save as a new video.
    
    Args:
        video_path (str): Path to the video file
        frame_number (int): Frame number to trim to
        output_dir (str): Directory to save the trimmed video
        
    Returns:
        str: Path to the trimmed video file
    """
    print(f"\n=== Trimming video to frame {frame_number} ===")
    if not video_path or not os.path.exists(video_path):
        print("❌ Error: Video file does not exist")
        return None
    
    try:
        # Get video information
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Failed to open video file")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate time in seconds
        time_seconds = frame_number / fps
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        timestamp = f"{int(time_seconds)}s"
        base_name = Path(video_path).stem
        output_file = os.path.join(output_dir, f"{base_name}_trimmed_to_{timestamp}.mp4")
        
        # Use ffmpeg to trim the video
        (
            ffmpeg
            .input(video_path)
            .output(output_file, to=time_seconds, c="copy")
            .global_args('-y')  # Overwrite output files
            .run(quiet=True)
        )
        
        if not os.path.exists(output_file):
            print("❌ Error: Failed to create trimmed video")
            return None
            
        print(f"✅ Successfully trimmed video to {time_seconds:.2f}s: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Error trimming video: {str(e)}")
        return None

def send_sharpest_frame_handler(gallery, selected_idx, frames_to_check=30):
    """
    Extract the sharpest frame from the last N frames of the selected video
    
    Args:
        gallery: Gradio gallery component with videos
        selected_idx: Index of the selected video
        frames_to_check: Number of frames from the end to check
        
    Returns:
        tuple: (image_path, video_path, frame_number, sharpness)
    """
    if gallery is None or not gallery:
        return None, None, None, "No videos in gallery"
        
    if selected_idx is None and len(gallery) == 1:
        selected_idx = 0
        
    if selected_idx is None or selected_idx >= len(gallery):
        return None, None, None, "No video selected"
    
    # Get the video path
    item = gallery[selected_idx]
    if isinstance(item, tuple):
        video_path = item[0]
    elif isinstance(item, dict):
        video_path = item.get('name') or item.get('data')
    else:
        video_path = str(item)
    
    # Extract the sharpest frame
    image_path, frame_number, sharpness = extract_sharpest_frame(video_path, frames_to_check)
    
    if image_path is None:
        return None, None, None, "Failed to extract sharpest frame"
    
    return image_path, video_path, frame_number, f"Extracted frame {frame_number} with sharpness {sharpness:.2f}"

def trim_and_prepare_for_extension(video_path, frame_number, save_path="outputs"):
    """
    Trim the video to the specified frame and prepare for extension.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame number to trim to
        save_path: Directory to save the trimmed video
        
    Returns:
        tuple: (trimmed_video_path, status_message)
    """
    if not video_path or not os.path.exists(video_path):
        return None, "No video selected or video file does not exist"
    
    if frame_number is None:
        return None, "No frame number provided, please extract sharpest frame first"
    
    # Trim the video
    trimmed_video = trim_video_to_frame(video_path, frame_number, save_path)
    
    if trimmed_video is None:
        return None, "Failed to trim video"
    
    return trimmed_video, f"Video trimmed to frame {frame_number} and ready for extension"

def send_last_frame_handler(gallery, selected_idx):
    """Handle sending last frame to input with better error handling"""
    if gallery is None or not gallery:
        return None, None
        
    if selected_idx is None and len(gallery) == 1:
        selected_idx = 0
        
    if selected_idx is None or selected_idx >= len(gallery):
        return None, None
        
    # Get the frame and video path
    frame = handle_last_frame_transfer(gallery, selected_idx)
    video_path = None
    
    if selected_idx < len(gallery):
        item = gallery[selected_idx]
        video_path = parse_video_path(item)
        
    return frame, video_path

def extract_last_frame(video_path: str) -> Optional[str]:
    """Extract last frame from video and return temporary image path with error handling"""
    print(f"\n=== Starting frame extraction ===")
    print(f"Input video path: {video_path}")
    
    if not video_path or not os.path.exists(video_path):
        print("❌ Error: Video file does not exist")
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Failed to open video file")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames detected: {total_frames}")
        
        if total_frames < 1:
            print("❌ Error: Video contains 0 frames")
            return None

        # Extract last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        success, frame = cap.read()
        
        if not success or frame is None:
            print("❌ Error: Failed to read last frame")
            return None

        # Prepare output path
        temp_dir = os.path.abspath("temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"last_frame_{os.path.basename(video_path)}.png")
        print(f"Saving frame to: {temp_path}")

        # Write and verify
        if not cv2.imwrite(temp_path, frame):
            print("❌ Error: Failed to write frame to file")
            return None
            
        if not os.path.exists(temp_path):
            print("❌ Error: Output file not created")
            return None

        print("✅ Frame extraction successful")
        return temp_path

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None
    finally:
        if 'cap' in locals():
            cap.release()

def handle_last_frame_transfer(gallery: list, selected_idx: int) -> Optional[str]:
    """Improved frame transfer with video input validation"""
    try:
        if gallery is None or not gallery:
            raise ValueError("No videos generated yet")
            
        if selected_idx is None:
            # Auto-select last generated video if batch_size=1
            if len(gallery) == 1:
                selected_idx = 0
            else:
                raise ValueError("Please select a video first")
                
        if selected_idx >= len(gallery):
            raise ValueError("Invalid selection index")
            
        item = gallery[selected_idx]
        
        # Video file existence check
        video_path = parse_video_path(item)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file missing: {video_path}")
            
        return extract_last_frame(video_path)
        
    except Exception as e:
        print(f"Frame transfer failed: {str(e)}")
        return None

def parse_video_path(item) -> str:
    """Parse different gallery item formats"""
    if isinstance(item, tuple):
        return item[0]
    elif isinstance(item, dict):
        return item.get('name') or item.get('data')
    return str(item)

def get_random_image_from_folder(folder_path):
    """Get a random image from the specified folder"""
    if not os.path.isdir(folder_path):
        return None, f"Error: {folder_path} is not a valid directory"
    
    # Get all image files in the folder
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    for ext in ('*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP'):
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        return None, f"Error: No image files found in {folder_path}"
    
    # Select a random image
    random_image = random.choice(image_files)
    return random_image, f"Selected: {os.path.basename(random_image)}"

def resize_image_keeping_aspect_ratio(image_path, max_width, max_height):
    """Resize image keeping aspect ratio and ensuring dimensions are divisible by 16"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = min(max_width, width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_height, height)
            new_width = int(new_height * aspect_ratio)
        
        # Make dimensions divisible by 16
        new_width = math.floor(new_width / 16) * 16
        new_height = math.floor(new_height / 16) * 16
        
        # Ensure minimum size
        new_width = max(16, new_width)
        new_height = max(16, new_height)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save to temporary file
        temp_path = f"temp_resized_{os.path.basename(image_path)}"
        resized_img.save(temp_path)
        
        return temp_path, (new_width, new_height)
    except Exception as e:
        return None, f"Error: {str(e)}"
# Function to process a batch of images from a folder
def batch_handler(
    use_random,
    prompt, negative_prompt, 
    width, height, 
    video_length, fps, infer_steps, 
    seed, flow_shift, guidance_scale, embedded_cfg_scale,
    batch_size, input_folder_path,
    dit_folder, model, vae, te1, te2, save_path, output_type, attn_mode, 
    block_swap, exclude_single_blocks, use_split_attn, use_fp8, split_uncond,
    lora_folder, *lora_params
):
    """Handle both folder-based batch processing and regular batch processing"""
    global stop_event
    
    # Check if this is a SkyReels model that needs special handling
    is_skyreels = "skyreels" in model.lower()
    is_skyreels_i2v = is_skyreels and "i2v" in model.lower()
    
    if use_random:
        # Random image from folder mode
        stop_event.clear()

        all_videos = []
        progress_text = "Starting generation..."
        yield [], "Preparing...", progress_text

        for i in range(batch_size):
            if stop_event.is_set():
                break

            batch_text = f"Generating video {i + 1} of {batch_size}"
            yield all_videos.copy(), batch_text, progress_text

            # Get random image from folder
            random_image, status = get_random_image_from_folder(input_folder_path)
            if random_image is None:
                yield all_videos, f"Error in batch {i+1}: {status}", ""
                continue

            # Resize image
            resized_image, size_info = resize_image_keeping_aspect_ratio(random_image, width, height)
            if resized_image is None:
                yield all_videos, f"Error resizing image in batch {i+1}: {size_info}", ""
                continue

            # If we have dimensions, update them
            local_width, local_height = width, height
            if isinstance(size_info, tuple):
                local_width, local_height = size_info
                progress_text = f"Using image: {os.path.basename(random_image)} - Resized to {local_width}x{local_height}"
            else:
                progress_text = f"Using image: {os.path.basename(random_image)}"
            
            yield all_videos.copy(), batch_text, progress_text

            # Calculate seed for this batch item
            current_seed = seed
            if seed == -1:
                current_seed = random.randint(0, 2**32 - 1)
            elif batch_size > 1:
                current_seed = seed + i

            # Process the image
            # For SkyReels models, we need to create a command with dit_in_channels=32
            if is_skyreels_i2v:
                env = os.environ.copy()
                env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
                env["PYTHONIOENCODING"] = "utf-8"
                
                model_path = os.path.join(dit_folder, model) if not os.path.isabs(model) else model
                
                # Extract parameters from lora_params
                num_lora_weights = 4
                lora_weights = lora_params[:num_lora_weights]
                lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
                
                cmd = [
                    sys.executable,
                    "hv_generate_video.py",
                    "--dit", model_path,
                    "--vae", vae,
                    "--text_encoder1", te1,
                    "--text_encoder2", te2,
                    "--prompt", prompt,
                    "--video_size", str(local_height), str(local_width),
                    "--video_length", str(video_length),
                    "--fps", str(fps),
                    "--infer_steps", str(infer_steps),
                    "--save_path", save_path,
                    "--seed", str(current_seed),
                    "--flow_shift", str(flow_shift),
                    "--embedded_cfg_scale", str(embedded_cfg_scale),
                    "--output_type", output_type,
                    "--attn_mode", attn_mode,
                    "--blocks_to_swap", str(block_swap),
                    "--fp8_llm",
                    "--vae_chunk_size", "32",
                    "--vae_spatial_tile_sample_min_size", "128",
                    "--dit_in_channels", "32",  # This is crucial for SkyReels i2v
                    "--image_path", resized_image  # Pass the image directly
                ]
                
                if use_fp8:
                    cmd.append("--fp8")
                
                if split_uncond:
                    cmd.append("--split_uncond")
                
                if use_split_attn:
                    cmd.append("--split_attn")
                
                if exclude_single_blocks:
                    cmd.append("--exclude_single_blocks")
                
                if negative_prompt:
                    cmd.extend(["--negative_prompt", negative_prompt])
                    
                if guidance_scale is not None:
                    cmd.extend(["--guidance_scale", str(guidance_scale)])
                
                # Add LoRA weights and multipliers if provided
                valid_loras = []
                for weight, mult in zip(lora_weights, lora_multipliers):
                    if weight and weight != "None":
                        valid_loras.append((os.path.join(lora_folder, weight), mult))
                
                if valid_loras:
                    weights = [weight for weight, _ in valid_loras]
                    multipliers = [str(mult) for _, mult in valid_loras]
                    cmd.extend(["--lora_weight"] + weights)
                    cmd.extend(["--lora_multiplier"] + multipliers)
                
                print(f"Running command: {' '.join(cmd)}")
                
                # Run the process
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1
                )
                
                while True:
                    if stop_event.is_set():
                        p.terminate()
                        p.wait()
                        yield all_videos, "Generation stopped by user.", ""
                        return
                        
                    line = p.stdout.readline()
                    if not line:
                        if p.poll() is not None:
                            break
                        continue
                        
                    print(line, end='')
                    if '|' in line and '%' in line and '[' in line and ']' in line:
                        yield all_videos.copy(), f"Processing video {i+1} (seed: {current_seed})", line.strip()
                
                p.stdout.close()
                p.wait()
                
                # Collect generated video
                save_path_abs = os.path.abspath(save_path)
                if os.path.exists(save_path_abs):
                    all_videos_files = sorted(
                        [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
                        key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
                        reverse=True
                    )
                    matching_videos = [v for v in all_videos_files if f"_{current_seed}" in v]
                    if matching_videos:
                        video_path = os.path.join(save_path_abs, matching_videos[0])
                        all_videos.append((str(video_path), f"Seed: {current_seed}"))
            else:
                # For non-SkyReels models, use the regular process_single_video function
                num_lora_weights = 4
                lora_weights = lora_params[:num_lora_weights]
                lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
                
                single_video_args = [
                    prompt, local_width, local_height, 1, video_length, fps, infer_steps,
                    current_seed, dit_folder, model, vae, te1, te2, save_path, flow_shift, embedded_cfg_scale,
                    output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
                    lora_folder
                ]
                single_video_args.extend(lora_weights)
                single_video_args.extend(lora_multipliers)
                single_video_args.extend([None, resized_image, None, negative_prompt, embedded_cfg_scale, split_uncond, guidance_scale, use_fp8])

                for videos, status, progress in process_single_video(*single_video_args):
                    if videos:
                        all_videos.extend(videos)
                    yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress

            # Clean up temporary file
            try:
                if os.path.exists(resized_image):
                    os.remove(resized_image)
            except:
                pass
            
            # Clear CUDA cache between generations
            clear_cuda_cache()
            time.sleep(0.5)

        yield all_videos, "Batch complete", ""
    else:
        # Regular image input - this is the part we need to fix
        # When a SkyReels I2V model is used, we need to use the direct command approach
        # with dit_in_channels=32 explicitly specified, just like in the folder processing branch
        if is_skyreels_i2v:
            stop_event.clear()
            
            all_videos = []
            progress_text = "Starting generation..."
            yield [], "Preparing...", progress_text
            
            # Extract lora parameters
            num_lora_weights = 4
            lora_weights = lora_params[:num_lora_weights]
            lora_multipliers = lora_params[num_lora_weights:num_lora_weights*2]
            extra_args = list(lora_params[num_lora_weights*2:]) if len(lora_params) > num_lora_weights*2 else []
            
            # Print extra_args for debugging
            print(f"Extra args: {extra_args}")
            
            # Get input image path from extra args - this is where we need to fix
            # In skyreels_generate_btn.click, we're passing skyreels_input which
            # should be the image path
            image_path = None
            if len(extra_args) > 0 and extra_args[0] is not None:
                image_path = extra_args[0]
                print(f"Image path found in extra_args[0]: {image_path}")
            
            # If we still don't have an image path, this is a problem
            if not image_path:
                # Let's try to debug what's happening - in the future, you can remove these
                # debug prints once everything works correctly
                print("No image path found in extra_args[0]")
                print(f"Full lora_params: {lora_params}")
                yield [], "Error: No input image provided", "An input image is required for SkyReels I2V models"
                return
            
            for i in range(batch_size):
                if stop_event.is_set():
                    yield all_videos, "Generation stopped by user", ""
                    return
                
                # Calculate seed for this batch item
                current_seed = seed
                if seed == -1:
                    current_seed = random.randint(0, 2**32 - 1)
                elif batch_size > 1:
                    current_seed = seed + i
                
                batch_text = f"Generating video {i + 1} of {batch_size}"
                yield all_videos.copy(), batch_text, progress_text
                
                # Set up environment
                env = os.environ.copy()
                env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
                env["PYTHONIOENCODING"] = "utf-8"
                
                model_path = os.path.join(dit_folder, model) if not os.path.isabs(model) else model
                
                # Build the command with dit_in_channels=32
                cmd = [
                    sys.executable,
                    "hv_generate_video.py",
                    "--dit", model_path,
                    "--vae", vae,
                    "--text_encoder1", te1,
                    "--text_encoder2", te2,
                    "--prompt", prompt,
                    "--video_size", str(height), str(width),
                    "--video_length", str(video_length),
                    "--fps", str(fps),
                    "--infer_steps", str(infer_steps),
                    "--save_path", save_path,
                    "--seed", str(current_seed),
                    "--flow_shift", str(flow_shift),
                    "--embedded_cfg_scale", str(embedded_cfg_scale),
                    "--output_type", output_type,
                    "--attn_mode", attn_mode,
                    "--blocks_to_swap", str(block_swap),
                    "--fp8_llm",
                    "--vae_chunk_size", "32",
                    "--vae_spatial_tile_sample_min_size", "128",
                    "--dit_in_channels", "32",  # This is crucial for SkyReels i2v
                    "--image_path", image_path
                ]
                
                if use_fp8:
                    cmd.append("--fp8")
                
                if split_uncond:
                    cmd.append("--split_uncond")
                
                if use_split_attn:
                    cmd.append("--split_attn")
                
                if exclude_single_blocks:
                    cmd.append("--exclude_single_blocks")
                
                if negative_prompt:
                    cmd.extend(["--negative_prompt", negative_prompt])
                    
                if guidance_scale is not None:
                    cmd.extend(["--guidance_scale", str(guidance_scale)])
                
                # Add LoRA weights and multipliers if provided
                valid_loras = []
                for weight, mult in zip(lora_weights, lora_multipliers):
                    if weight and weight != "None":
                        valid_loras.append((os.path.join(lora_folder, weight), mult))
                
                if valid_loras:
                    weights = [weight for weight, _ in valid_loras]
                    multipliers = [str(mult) for _, mult in valid_loras]
                    cmd.extend(["--lora_weight"] + weights)
                    cmd.extend(["--lora_multiplier"] + multipliers)
                
                print(f"Running command: {' '.join(cmd)}")
                
                # Run the process
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1
                )
                
                while True:
                    if stop_event.is_set():
                        p.terminate()
                        p.wait()
                        yield all_videos, "Generation stopped by user.", ""
                        return
                        
                    line = p.stdout.readline()
                    if not line:
                        if p.poll() is not None:
                            break
                        continue
                        
                    print(line, end='')
                    if '|' in line and '%' in line and '[' in line and ']' in line:
                        yield all_videos.copy(), f"Processing (seed: {current_seed})", line.strip()
                
                p.stdout.close()
                p.wait()
                
                # Collect generated video
                save_path_abs = os.path.abspath(save_path)
                if os.path.exists(save_path_abs):
                    all_videos_files = sorted(
                        [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
                        key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
                        reverse=True
                    )
                    matching_videos = [v for v in all_videos_files if f"_{current_seed}" in v]
                    if matching_videos:
                        video_path = os.path.join(save_path_abs, matching_videos[0])
                        all_videos.append((str(video_path), f"Seed: {current_seed}"))
                
                # Clear CUDA cache between generations
                clear_cuda_cache()
                time.sleep(0.5)
            
            yield all_videos, "Batch complete", ""
        else:
            # For regular non-SkyReels models, use the original process_batch function
            regular_args = [
                prompt, width, height, batch_size, video_length, fps, infer_steps,
                seed, dit_folder, model, vae, te1, te2, save_path, flow_shift, guidance_scale,
                output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
                lora_folder
            ]
            yield from process_batch(*(regular_args + list(lora_params)))

def get_dit_models(dit_folder: str) -> List[str]:
    """Get list of available DiT models in the specified folder"""
    if not os.path.exists(dit_folder):
        return ["mp_rank_00_model_states.pt"]
    models = [f for f in os.listdir(dit_folder) if f.endswith('.pt') or f.endswith('.safetensors')]
    models.sort(key=str.lower)
    return models if models else ["mp_rank_00_model_states.pt"]

def update_dit_and_lora_dropdowns(dit_folder: str, lora_folder: str, *current_values) -> List[gr.update]:
    """Update both DiT and LoRA dropdowns"""
    # Get model lists
    dit_models = get_dit_models(dit_folder)
    lora_choices = get_lora_options(lora_folder)
    
    # Current values processing
    dit_value = current_values[0]
    if dit_value not in dit_models:
        dit_value = dit_models[0] if dit_models else None
        
    weights = current_values[1:5]
    multipliers = current_values[5:9]
    
    results = [gr.update(choices=dit_models, value=dit_value)]
    
    # Add LoRA updates
    for i in range(4):
        weight = weights[i] if i < len(weights) else "None"
        multiplier = multipliers[i] if i < len(multipliers) else 1.0
        if weight not in lora_choices:
            weight = "None"
        results.extend([
            gr.update(choices=lora_choices, value=weight),
            gr.update(value=multiplier)
        ])
    
    return results

def extract_video_metadata(video_path: str) -> Dict:
    """Extract metadata from video file using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        metadata = json.loads(result.stdout.decode('utf-8'))
        if 'format' in metadata and 'tags' in metadata['format']:
            comment = metadata['format']['tags'].get('comment', '{}')
            return json.loads(comment)
        return {}
    except Exception as e:
        print(f"Metadata extraction failed: {str(e)}")
        return {}

def create_parameter_transfer_map(metadata: Dict, target_tab: str) -> Dict:
    """Map metadata parameters to Gradio components for different tabs"""
    mapping = {
        'common': {
            'prompt': ('prompt', 'v2v_prompt', 'wanx_v2v_prompt'),  # Add WanX-v2v mapping
            'width': ('width', 'v2v_width', 'wanx_v2v_width'),
            'height': ('height', 'v2v_height', 'wanx_v2v_height'),
            'batch_size': ('batch_size', 'v2v_batch_size', 'wanx_v2v_batch_size'),
            'video_length': ('video_length', 'v2v_video_length', 'wanx_v2v_video_length'),
            'fps': ('fps', 'v2v_fps', 'wanx_v2v_fps'),
            'infer_steps': ('infer_steps', 'v2v_infer_steps', 'wanx_v2v_infer_steps'),
            'seed': ('seed', 'v2v_seed', 'wanx_v2v_seed'),
            'flow_shift': ('flow_shift', 'v2v_flow_shift', 'wanx_v2v_flow_shift'),
            'guidance_scale': ('cfg_scale', 'v2v_cfg_scale', 'wanx_v2v_guidance_scale'),
            'negative_prompt': ('negative_prompt', 'v2v_negative_prompt', 'wanx_v2v_negative_prompt'),
            'strength': ('strength', 'v2v_strength', 'wanx_v2v_strength')
        },
        'lora': {
            'lora_weights': [(f'lora{i+1}', f'v2v_lora_weights[{i}]', f'wanx_v2v_lora_weights[{i}]') for i in range(4)],
            'lora_multipliers': [(f'lora{i+1}_multiplier', f'v2v_lora_multipliers[{i}]', f'wanx_v2v_lora_multipliers[{i}]') for i in range(4)]
        }
    }
    
    results = {}
    for param, value in metadata.items():
        # Handle common parameters
        if param in mapping['common']:
            target_idx = 0 if target_tab == 't2v' else 1 if target_tab == 'v2v' else 2
            if target_idx < len(mapping['common'][param]):
                target = mapping['common'][param][target_idx]
                results[target] = value
        
        # Handle LoRA parameters
        if param == 'lora_weights':
            for i, weight in enumerate(value[:4]):
                target_idx = 0 if target_tab == 't2v' else 1 if target_tab == 'v2v' else 2
                if target_idx < len(mapping['lora']['lora_weights'][i]):
                    target = mapping['lora']['lora_weights'][i][target_idx]
                    results[target] = weight
                
        if param == 'lora_multipliers':
            for i, mult in enumerate(value[:4]):
                target_idx = 0 if target_tab == 't2v' else 1 if target_tab == 'v2v' else 2
                if target_idx < len(mapping['lora']['lora_multipliers'][i]):
                    target = mapping['lora']['lora_multipliers'][i][target_idx]
                    results[target] = float(mult)
                
    return results

def add_metadata_to_video(video_path: str, parameters: dict) -> None:
    """Add generation parameters to video metadata using ffmpeg."""
    import json
    import subprocess

    # Convert parameters to JSON string
    params_json = json.dumps(parameters, indent=2)
    
    # Temporary output path
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    
    # Add Fun-Control information to metadata if applicable
    task = parameters.get("task", "")
    if task.endswith("-FC"):
        parameters["fun_control"] = True
        # Store the control path in metadata if available
        if "control_path" in parameters:
            parameters["control_video"] = os.path.basename(parameters["control_path"])
    
    # FFmpeg command to add metadata without re-encoding
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-metadata', f'comment={params_json}',
        '-codec', 'copy',
        temp_path
    ]
    
    try:
        # Execute FFmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Replace original file with the metadata-enhanced version
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError as e:
        print(f"Failed to add metadata: {e.stderr.decode()}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Error: {str(e)}")

def count_prompt_tokens(prompt: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(prompt)
    return len(tokens)


def get_lora_options(lora_folder: str = "lora") -> List[str]:
    if not os.path.exists(lora_folder):
        return ["None"]
    lora_files = [f for f in os.listdir(lora_folder) if f.endswith('.safetensors') or f.endswith('.pt')]
    lora_files.sort(key=str.lower)
    return ["None"] + lora_files

def update_lora_dropdowns(lora_folder: str, *current_values) -> List[gr.update]:
    new_choices = get_lora_options(lora_folder)
    weights = current_values[:4]
    multipliers = current_values[4:8]
    
    results = []
    for i in range(4):
        weight = weights[i] if i < len(weights) else "None"
        multiplier = multipliers[i] if i < len(multipliers) else 1.0
        if weight not in new_choices:
            weight = "None"
        results.extend([
            gr.update(choices=new_choices, value=weight),
            gr.update(value=multiplier) 
        ])
    
    return results

def send_to_v2v(evt: gr.SelectData, gallery: list, prompt: str, selected_index: gr.State) -> Tuple[Optional[str], str, int]:
    """Transfer selected video and prompt to Video2Video tab"""
    if not gallery or evt.index >= len(gallery):
        return None, "", selected_index.value
    
    selected_item = gallery[evt.index]
    
    # Handle different gallery item formats
    if isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    elif isinstance(selected_item, (tuple, list)):
        video_path = selected_item[0]
    else:
        video_path = selected_item
    
    # Final cleanup for Gradio Video component
    if isinstance(video_path, tuple):
        video_path = video_path[0]
    
    # Update the selected index
    selected_index.value = evt.index
    
    return str(video_path), prompt, evt.index

def send_selected_to_v2v(gallery: list, prompt: str, selected_index: gr.State) -> Tuple[Optional[str], str]:
    """Send the currently selected video to V2V tab"""
    if not gallery or selected_index.value is None or selected_index.value >= len(gallery):
        return None, ""
    
    selected_item = gallery[selected_index.value]
    
    # Handle different gallery item formats
    if isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    elif isinstance(selected_item, (tuple, list)):
        video_path = selected_item[0]
    else:
        video_path = selected_item
    
    # Final cleanup for Gradio Video component
    if isinstance(video_path, tuple):
        video_path = video_path[0]
    
    return str(video_path), prompt

def clear_cuda_cache():
    """Clear CUDA cache if available"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Optional: synchronize to ensure cache is cleared
        torch.cuda.synchronize()

def wanx_batch_handler(
    use_random,
    prompt, 
    negative_prompt,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    seed,
    batch_size,
    input_folder_path,
    wanx_input_end,
    task,
    dit_folder,
    dit_path,
    vae_path,
    t5_path,
    clip_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers: str,
    slg_start: Optional[str],
    slg_end: Optional[str],
    enable_cfg_skip: bool,
    cfg_skip_mode: str,
    cfg_apply_ratio: float,
    enable_preview: bool,
    preview_steps: int,
    *lora_params,   # <-- DO NOT ADD NAMED ARGS AFTER THIS!
):
    """Handle both folder-based batch processing and regular processing for all WanX tabs"""
    global stop_event
    
    # Convert None strings to actual None
    slg_layers = None if slg_layers == "None" else slg_layers
    slg_start = None if slg_start == "None" else slg_start
    slg_end = None if slg_end == "None" else slg_end
    
    # Construct full dit_path including folder
    full_dit_path = os.path.join(dit_folder, dit_path) if not os.path.isabs(dit_path) else dit_path    
    # Clean up LoRA params to proper format
    clean_lora_params = []
    for param in lora_params:
        # Convert None strings to "None" for consistency
        if param is None or str(param).lower() == "none":
            clean_lora_params.append("None")
        else:
            clean_lora_params.append(str(param))
    
    # Extract LoRA weights and multipliers
    num_lora_weights = 4
    lora_weights = clean_lora_params[:num_lora_weights]
    lora_multipliers = []
    for mult in clean_lora_params[num_lora_weights:num_lora_weights*2]:
        try:
            lora_multipliers.append(float(mult))
        except (ValueError, TypeError):
            lora_multipliers.append(1.0)
    while len(lora_weights) < 4:
        lora_weights.append("None")
    while len(lora_multipliers) < 4:
        lora_multipliers.append(1.0)

    # Now extract trailing params: input_file, control_video, control_strength, control_start, control_end
    remaining_params = clean_lora_params[num_lora_weights*2:]
    input_file = remaining_params[0] if len(remaining_params) > 0 else None
    control_video = remaining_params[1] if len(remaining_params) > 1 else None
    try:
        control_strength = float(remaining_params[2]) if len(remaining_params) > 2 else 1.0
    except Exception:
        control_strength = 1.0
    try:
        control_start = float(remaining_params[3]) if len(remaining_params) > 3 else 0.0
    except Exception:
        control_start = 0.0
    try:
        control_end = float(remaining_params[4]) if len(remaining_params) > 4 else 1.0
    except Exception:
        control_end = 1.0

    yield [], [], "Preparing batch...", "" # Clear main and preview galleries

    if use_random:
        stop_event.clear()
        all_videos = []
        all_previews = [] # Keep track of previews from the last successful item? Or clear each time? Let's clear.
        progress_text = "Starting generation..."
        yield [], [], "Preparing...", progress_text # Clear galleries again just in case
        batch_size = int(batch_size)
        for i in range(batch_size):
            if stop_event.is_set():
                yield all_videos, [], "Generation stopped by user", "" # Yield empty previews on stop
                return

            # --- Clear previews for this item ---
            current_previews_for_item = []
            yield all_videos.copy(), current_previews_for_item, f"Generating video {i + 1} of {batch_size}", progress_text # Yield cleared previews

            # ... (Keep existing random image logic: get random, resize) ...
            random_image, status = get_random_image_from_folder(input_folder_path)
            if random_image is None:
                yield all_videos, current_previews_for_item, f"Error in batch {i+1}: {status}", ""
                continue # Skip to next batch item on error

            resized_image, size_info = resize_image_keeping_aspect_ratio(random_image, width, height)
            if resized_image is None:
                yield all_videos, current_previews_for_item, f"Error resizing image in batch {i+1}: {size_info}", ""
                # Clean up the random image if resize failed but image exists
                try:
                    if os.path.exists(random_image) and "temp_resized" not in random_image: # Avoid double delete if resize output existed
                       pass # Might not want to delete original random image here
                except: pass
                continue # Skip to next batch item on error

            local_width, local_height = width, height
            if isinstance(size_info, tuple): local_width, local_height = size_info
            progress_text = f"Using image: {os.path.basename(random_image)} - Resized to {local_width}x{local_height}"
            yield all_videos.copy(), current_previews_for_item, f"Generating video {i + 1} of {batch_size}", progress_text

            current_seed = seed
            if seed == -1: current_seed = random.randint(0, 2**32 - 1)
            elif batch_size > 1: current_seed = seed + i

            # --- Corrected call to wanx_generate_video with accumulation ---
            newly_generated_video = None # Track the video generated *in this iteration*
            last_status_for_item = f"Generating video {i+1}/{batch_size}" # Keep track of last status
            last_progress_for_item = progress_text # Keep track of last progress line

            # Inner loop iterates through the generator for ONE batch item
            for videos_update, previews_update, status, progress in wanx_generate_video(
                prompt, negative_prompt, resized_image, local_width, local_height,
                video_length, fps, infer_steps, flow_shift, guidance_scale, current_seed,
                wanx_input_end, # Pass the argument
                task, dit_folder, full_dit_path, vae_path, t5_path, clip_path, save_path,
                output_type, sample_solver, exclude_single_blocks, attn_mode, block_swap,
                fp8, fp8_scaled, fp8_t5, lora_folder,
                slg_layers, slg_start, slg_end,
                lora_weights[0], lora_weights[1], lora_weights[2], lora_weights[3],
                lora_multipliers[0], lora_multipliers[1], lora_multipliers[2], lora_multipliers[3],
                enable_cfg_skip, cfg_skip_mode, cfg_apply_ratio,
                None, 1.0, 0.0, 1.0, # Placeholders for control video args in random mode
                enable_preview=enable_preview,
                preview_steps=preview_steps
            ):
                # Store the latest video info from this *specific* generator run
                if videos_update:
                    # wanx_generate_video yields the *full* list it knows about,
                    # so we take the last item assuming it's the new one.
                    newly_generated_video = videos_update[-1]

                current_previews_for_item = previews_update # Update previews for *this* item
                last_status_for_item = f"Batch {i+1}/{batch_size}: {status}" # Store last status
                last_progress_for_item = progress # Store last progress line
                # Yield the *current cumulative* list during progress updates
                yield all_videos.copy(), current_previews_for_item, last_status_for_item, last_progress_for_item

            # --- After the inner loop finishes for item 'i' ---
            # Now, add the video generated in this iteration to the main list
            if newly_generated_video and newly_generated_video not in all_videos:
                all_videos.append(newly_generated_video)
                print(f"DEBUG: Appended video {newly_generated_video[1] if isinstance(newly_generated_video, tuple) else 'unknown'} to all_videos (Total: {len(all_videos)})")
                # Yield the updated cumulative list *immediately* after appending
                yield all_videos.copy(), current_previews_for_item, last_status_for_item, last_progress_for_item
            elif not newly_generated_video:
                 print(f"DEBUG: No new video generated or yielded by wanx_generate_video for batch item {i+1}.")


            # --- Cleanup for item 'i' (Correctly indented) ---
            try:
                # Only remove the temporary resized image
                if os.path.exists(resized_image) and "temp_resized" in resized_image:
                     os.remove(resized_image)
                     print(f"DEBUG: Removed temporary resized image: {resized_image}")
            except Exception as e:
                print(f"Warning: Could not remove temp image {resized_image}: {e}")
            clear_cuda_cache()
            time.sleep(0.5)
            # --- End Cleanup for item 'i' ---

        # --- After the outer loop (all batch items processed) ---
        yield all_videos, [], "Batch complete", "" # Yield empty previews at the end
    else:
        # ... (Keep existing checks for non-random mode: input file, control video) ...
        batch_size = int(batch_size)
        if not input_file and "i2v" in task:
            yield [], [], "Error: No input image provided", "An input image is required for I2V models"
            return
        if "-FC" in task and not control_video:
            yield [], [], "Error: No control video provided", "A control video is required for Fun-Control models"
            return

        if batch_size > 1:
            stop_event.clear()
            all_videos = []
            all_previews = [] # Clear previews at start of batch
            progress_text = "Starting generation..."
            yield [], [], "Preparing...", progress_text # Clear galleries

            for i in range(batch_size):
                if stop_event.is_set():
                    yield all_videos, [], "Generation stopped by user", "" # Yield empty previews
                    return

                # --- Clear previews for this item ---
                current_previews_for_item = []
                yield all_videos.copy(), current_previews_for_item, f"Generating video {i+1}/{batch_size}", progress_text

                current_seed = seed
                if seed == -1: current_seed = random.randint(0, 2**32 - 1)
                elif batch_size > 1: current_seed = seed + i
                batch_text = f"Generating video {i+1}/{batch_size} (seed: {current_seed})"
                yield all_videos.copy(), current_previews_for_item, batch_text, progress_text # Update status

                # --- Corrected call to wanx_generate_video with accumulation ---
                newly_generated_video = None # Track the video generated *in this iteration*
                last_status_for_item = f"Generating video {i+1}/{batch_size}" # Keep track of last status
                last_progress_for_item = progress_text # Keep track of last progress line

                # Inner loop iterates through the generator for ONE batch item
                for videos_update, previews_update, status, progress in wanx_generate_video(
                    prompt, negative_prompt, input_file, width, height,
                    video_length, fps, infer_steps, flow_shift, guidance_scale, current_seed,
                    wanx_input_end, # Pass the argument
                    task, dit_folder, full_dit_path, vae_path, t5_path, clip_path, save_path,
                    output_type, sample_solver, exclude_single_blocks, attn_mode, block_swap,
                    fp8, fp8_scaled, fp8_t5, lora_folder,
                    slg_layers, slg_start, slg_end,
                    lora_weights[0], lora_weights[1], lora_weights[2], lora_weights[3],
                    lora_multipliers[0], lora_multipliers[1], lora_multipliers[2], lora_multipliers[3],
                    enable_cfg_skip, cfg_skip_mode, cfg_apply_ratio,
                    control_video, control_strength, control_start, control_end,
                    # --- Pass preview args ---
                    enable_preview=enable_preview,
                    preview_steps=preview_steps
                ):
                     # Store the latest video info from this *specific* generator run
                    if videos_update:
                        # wanx_generate_video yields the *full* list it knows about,
                        # so we take the last item assuming it's the new one.
                        newly_generated_video = videos_update[-1]

                    current_previews_for_item = previews_update # Update previews for *this* item
                    last_status_for_item = f"Batch {i+1}/{batch_size}: {status}" # Store last status
                    last_progress_for_item = progress # Store last progress line
                    # Yield the *current cumulative* list during progress updates
                    yield all_videos.copy(), current_previews_for_item, last_status_for_item, last_progress_for_item

                # --- After the inner loop finishes for item 'i' ---
                # Now, add the video generated in this iteration to the main list
                if newly_generated_video and newly_generated_video not in all_videos:
                    all_videos.append(newly_generated_video)
                    print(f"DEBUG: Appended video {newly_generated_video[1] if isinstance(newly_generated_video, tuple) else 'unknown'} to all_videos (Total: {len(all_videos)})")
                    # Yield the updated cumulative list *immediately* after appending
                    yield all_videos.copy(), current_previews_for_item, last_status_for_item, last_progress_for_item
                elif not newly_generated_video:
                    print(f"DEBUG: No new video generated or yielded by wanx_generate_video for batch item {i+1}.")
                # --- End modified call ---

                clear_cuda_cache()
                time.sleep(0.5)
            yield all_videos, [], "Batch complete", "" # Yield empty previews at the end
        else: # Single generation (batch_size = 1)
            stop_event.clear()
            # --- Modified call to wanx_generate_video (yield from) ---
            # Add preview args directly
            yield from wanx_generate_video(
                prompt, negative_prompt, input_file, width, height,
                video_length, fps, infer_steps, flow_shift, guidance_scale, seed,
                wanx_input_end, # Pass the argument
                task, dit_folder, full_dit_path, vae_path, t5_path, clip_path, save_path,
                output_type, sample_solver, exclude_single_blocks, attn_mode, block_swap,
                fp8, fp8_scaled, fp8_t5, lora_folder,
                slg_layers, slg_start, slg_end,
                lora_weights[0], lora_weights[1], lora_weights[2], lora_weights[3],
                lora_multipliers[0], lora_multipliers[1], lora_multipliers[2], lora_multipliers[3],
                enable_cfg_skip, cfg_skip_mode, cfg_apply_ratio,
                control_video, control_strength, control_start, control_end,
                # --- Pass preview args ---
                enable_preview=enable_preview,
                preview_steps=preview_steps
            )

def process_single_video(
    prompt: str,
    width: int,
    height: int,
    batch_size: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    dit_folder: str,
    model: str,
    vae: str,
    te1: str,
    te2: str,
    save_path: str,
    flow_shift: float,
    cfg_scale: float,
    output_type: str,
    attn_mode: str,
    block_swap: int,
    exclude_single_blocks: bool,
    use_split_attn: bool,    
    lora_folder: str,
    lora1: str = "",
    lora2: str = "",
    lora3: str = "",
    lora4: str = "",
    lora1_multiplier: float = 1.0,
    lora2_multiplier: float = 1.0,
    lora3_multiplier: float = 1.0,
    lora4_multiplier: float = 1.0,
    video_path: Optional[str] = None,
    image_path: Optional[str] = None,
    strength: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    embedded_cfg_scale: Optional[float] = None,
    split_uncond: Optional[bool] = None,
    guidance_scale: Optional[float] = None,
    use_fp8: bool = True
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate a single video with the given parameters"""
    global stop_event
    
    if stop_event.is_set():
        yield [], "", ""
        return

    # Determine if this is a SkyReels model and what type
    is_skyreels = "skyreels" in model.lower()
    is_skyreels_i2v = is_skyreels and "i2v" in model.lower()
    is_skyreels_t2v = is_skyreels and "t2v" in model.lower()
    
    if is_skyreels:
        # Force certain parameters for SkyReels
        if negative_prompt is None:
            negative_prompt = ""
        if embedded_cfg_scale is None:
            embedded_cfg_scale = 1.0  # Force to 1.0 for SkyReels
        if split_uncond is None:
            split_uncond = True
        if guidance_scale is None:
            guidance_scale = cfg_scale  # Use cfg_scale as guidance_scale if not provided
            
        # Determine the input channels based on model type
        if is_skyreels_i2v:
            dit_in_channels = 32  # SkyReels I2V uses 32 channels
        else:
            dit_in_channels = 16  # SkyReels T2V uses 16 channels (same as regular models)
    else:
        dit_in_channels = 16  # Regular Hunyuan models use 16 channels
        embedded_cfg_scale = cfg_scale 

    if os.path.isabs(model):
        model_path = model
    else:
        model_path = os.path.normpath(os.path.join(dit_folder, model))
    
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["BATCH_RUN_ID"] = f"{time.time()}"

    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)
    else:
        batch_id = int(env.get("BATCH_RUN_ID", "0").split('.')[-1])
        if batch_size > 1:  # Only modify seed for batch generation
            current_seed = (seed + batch_id * 100003) % (2**32)
        else:
            current_seed = seed

    clear_cuda_cache()

    command = [
        sys.executable,
        "hv_generate_video.py",
        "--dit", model_path,
        "--vae", vae,
        "--text_encoder1", te1,
        "--text_encoder2", te2,
        "--prompt", prompt,
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", save_path,
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--embedded_cfg_scale", str(cfg_scale),
        "--output_type", output_type,
        "--attn_mode", attn_mode,
        "--blocks_to_swap", str(block_swap),
        "--fp8_llm",
        "--vae_chunk_size", "32",
        "--vae_spatial_tile_sample_min_size", "128"
    ]
    
    if use_fp8:
        command.append("--fp8")

    # Add negative prompt and embedded cfg scale for SkyReels
    if is_skyreels:
        command.extend(["--dit_in_channels", str(dit_in_channels)])
        command.extend(["--guidance_scale", str(guidance_scale)])
        
        if negative_prompt:
            command.extend(["--negative_prompt", negative_prompt])
        if split_uncond:
            command.append("--split_uncond")

    # Add LoRA weights and multipliers if provided
    valid_loras = []
    for weight, mult in zip([lora1, lora2, lora3, lora4], 
                          [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]):
        if weight and weight != "None":
            valid_loras.append((os.path.join(lora_folder, weight), mult))
    if valid_loras:
        weights = [weight for weight, _ in valid_loras]
        multipliers = [str(mult) for _, mult in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)

    if exclude_single_blocks:
        command.append("--exclude_single_blocks")
    if use_split_attn:
        command.append("--split_attn")

    # Handle input paths
    if video_path:
        command.extend(["--video_path", video_path])
        if strength is not None:
            command.extend(["--strength", str(strength)])
    elif image_path:
        command.extend(["--image_path", image_path])
        # Only add strength parameter for non-SkyReels I2V models
        # SkyReels I2V doesn't use strength parameter for image-to-video generation
        if strength is not None and not is_skyreels_i2v:
            command.extend(["--strength", str(strength)])
            
    print(f"{command}")

    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = []
    
    while True:
        if stop_event.is_set():
            p.terminate()
            p.wait()
            yield [], "", "Generation stopped by user."
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue
            
        print(line, end='')
        if '|' in line and '%' in line and '[' in line and ']' in line:
            yield videos.copy(), f"Processing (seed: {current_seed})", line.strip()

    p.stdout.close()
    p.wait()

    clear_cuda_cache()
    time.sleep(0.5)

    # Collect generated video
    save_path_abs = os.path.abspath(save_path)
    if os.path.exists(save_path_abs):
        all_videos = sorted(
            [f for f in os.listdir(save_path_abs) if f.endswith('.mp4')],
            key=lambda x: os.path.getmtime(os.path.join(save_path_abs, x)),
            reverse=True
        )
        matching_videos = [v for v in all_videos if f"_{current_seed}" in v]
        if matching_videos:
            video_path = os.path.join(save_path_abs, matching_videos[0])
            
            # Collect parameters for metadata
            parameters = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "video_length": video_length,
                "fps": fps,
                "infer_steps": infer_steps,
                "seed": current_seed,
                "model": model,
                "vae": vae,
                "te1": te1,
                "te2": te2,
                "save_path": save_path,
                "flow_shift": flow_shift,
                "cfg_scale": cfg_scale,
                "output_type": output_type,
                "attn_mode": attn_mode,
                "block_swap": block_swap,
                "lora_weights": [lora1, lora2, lora3, lora4],
                "lora_multipliers": [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier],
                "input_video": video_path if video_path else None,
                "input_image": image_path if image_path else None,
                "strength": strength,
                "negative_prompt": negative_prompt if is_skyreels else None,
                "embedded_cfg_scale": embedded_cfg_scale if is_skyreels else None
            }
            
            add_metadata_to_video(video_path, parameters)
            videos.append((str(video_path), f"Seed: {current_seed}"))

    yield videos, f"Completed (seed: {current_seed})", ""

# The issue is in the process_batch function, in the section that handles different input types
# Here's the corrected version of that section:

def process_batch(
    prompt: str,
    width: int,
    height: int,
    batch_size: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    dit_folder: str,
    model: str,
    vae: str,
    te1: str,
    te2: str,
    save_path: str,
    flow_shift: float,
    cfg_scale: float,
    output_type: str,
    attn_mode: str,
    block_swap: int,
    exclude_single_blocks: bool,
    use_split_attn: bool,
    lora_folder: str,
    *args
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Process a batch of videos using Gradio's queue"""
    global stop_event
    stop_event.clear()

    all_videos = []
    progress_text = "Starting generation..."
    yield [], "Preparing...", progress_text

    # Extract additional arguments
    num_lora_weights = 4
    lora_weights = args[:num_lora_weights]
    lora_multipliers = args[num_lora_weights:num_lora_weights*2]
    extra_args = args[num_lora_weights*2:]

    # Determine if this is a SkyReels model and what type
    is_skyreels = "skyreels" in model.lower()
    is_skyreels_i2v = is_skyreels and "i2v" in model.lower()
    is_skyreels_t2v = is_skyreels and "t2v" in model.lower()

    # Handle input paths and additional parameters
    input_path = extra_args[0] if extra_args else None
    strength = float(extra_args[1]) if len(extra_args) > 1 else None
    
    # Get use_fp8 flag (it should be the last parameter)
    use_fp8 = bool(extra_args[-1]) if extra_args and len(extra_args) >= 3 else True
    
    # Get SkyReels specific parameters if applicable
    if is_skyreels:
        # Always set embedded_cfg_scale to 1.0 for SkyReels models
        embedded_cfg_scale = 1.0
        
        negative_prompt = str(extra_args[2]) if len(extra_args) > 2 and extra_args[2] is not None else ""
        # Use cfg_scale for guidance_scale parameter
        guidance_scale = float(extra_args[3]) if len(extra_args) > 3 and extra_args[3] is not None else cfg_scale
        split_uncond = True if len(extra_args) > 4 and extra_args[4] else False
    else:
        negative_prompt = str(extra_args[2]) if len(extra_args) > 2 and extra_args[2] is not None else None
        guidance_scale = cfg_scale
        embedded_cfg_scale = cfg_scale
        split_uncond = bool(extra_args[4]) if len(extra_args) > 4 else None

    for i in range(batch_size):
        if stop_event.is_set():
            break

        batch_text = f"Generating video {i + 1} of {batch_size}"
        yield all_videos.copy(), batch_text, progress_text

        # Handle different input types
        video_path = None
        image_path = None
        
        if input_path:
            # Check if it's an image file (common image extensions)
            is_image = False
            lower_path = input_path.lower()
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            is_image = any(lower_path.endswith(ext) for ext in image_extensions)
            
            # Only use image_path for SkyReels I2V models and actual image files
            if is_skyreels_i2v and is_image:
                image_path = input_path
            else:
                video_path = input_path

        # Prepare arguments for process_single_video
        single_video_args = [
            prompt, width, height, batch_size, video_length, fps, infer_steps,
            seed, dit_folder, model, vae, te1, te2, save_path, flow_shift, cfg_scale,
            output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
            lora_folder
        ]
        single_video_args.extend(lora_weights)
        single_video_args.extend(lora_multipliers)
        single_video_args.extend([video_path, image_path, strength, negative_prompt, embedded_cfg_scale, split_uncond, guidance_scale, use_fp8])

        for videos, status, progress in process_single_video(*single_video_args):
            if videos:
                all_videos.extend(videos)
            yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress

    yield all_videos, "Batch complete", ""

def update_wanx_image_dimensions(image):
    """Update dimensions from uploaded image"""
    if image is None:
        return "", gr.update(value=832), gr.update(value=480)
    img = Image.open(image)
    w, h = img.size
    w = (w // 32) * 32
    h = (h // 32) * 32
    return f"{w}x{h}", w, h

def calculate_wanx_width(height, original_dims):
    """Calculate width based on height maintaining aspect ratio"""
    if not original_dims:
        return gr.update()
    orig_w, orig_h = map(int, original_dims.split('x'))
    aspect_ratio = orig_w / orig_h
    new_width = math.floor((height * aspect_ratio) / 32) * 32
    return gr.update(value=new_width)

def calculate_wanx_height(width, original_dims):
    """Calculate height based on width maintaining aspect ratio"""
    if not original_dims:
        return gr.update()
    orig_w, orig_h = map(int, original_dims.split('x'))
    aspect_ratio = orig_w / orig_h
    new_height = math.floor((width / aspect_ratio) / 32) * 32
    return gr.update(value=new_height)

def update_wanx_from_scale(scale, original_dims):
    """Update dimensions based on scale percentage"""
    if not original_dims:
        return gr.update(), gr.update()
    orig_w, orig_h = map(int, original_dims.split('x'))
    new_w = math.floor((orig_w * scale / 100) / 32) * 32
    new_h = math.floor((orig_h * scale / 100) / 32) * 32
    return gr.update(value=new_w), gr.update(value=new_h)

def recommend_wanx_flow_shift(width, height):
    """Get recommended flow shift value based on dimensions"""
    recommended_shift = 3.0 if (width == 832 and height == 480) or (width == 480 and height == 832) else 5.0
    return gr.update(value=recommended_shift)

def handle_wanx_gallery_select(evt: gr.SelectData, gallery) -> tuple:
    """Track selected index and video path when gallery item is clicked"""
    if gallery is None:
        return None, None
    
    if evt.index >= len(gallery):
        return None, None
    
    selected_item = gallery[evt.index]
    video_path = None
    
    # Extract the video path based on the item type
    if isinstance(selected_item, tuple):
        video_path = selected_item[0]
    elif isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    else:
        video_path = selected_item
    
    return evt.index, video_path

def get_step_from_preview_path(path):
    match = re.search(r"step_(\d+)_", os.path.basename(path))
    return int(match.group(1)) if match else -1

def wanx_generate_video(
    prompt, 
    negative_prompt,
    input_image,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    seed,
    wanx_input_end,
    task,
    dit_folder,
    dit_path,
    vae_path,
    t5_path,
    clip_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers,
    slg_start,
    slg_end,
    lora1="None",
    lora2="None",
    lora3="None",
    lora4="None",
    lora1_multiplier=1.0,
    lora2_multiplier=1.0,
    lora3_multiplier=1.0,
    lora4_multiplier=1.0,
    enable_cfg_skip=False,
    cfg_skip_mode="none",
    cfg_apply_ratio=0.7,
    control_video=None,
    control_strength=1.0,
    control_start=0.0,
    control_end=1.0,
    enable_preview: bool = False,
    preview_steps: int = 5
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate video with WanX model (supports both i2v, t2v and Fun-Control)"""
    global stop_event

    current_previews = []
    yield [], current_previews, "Preparing...", "" # Yield empty previews
    
    # Fix 1: Ensure lora_folder is a string
    lora_folder = str(lora_folder) if lora_folder else "lora"
    
    # Debug prints
    print(f"DEBUG - LoRA params: {lora1}, {lora2}, {lora3}, {lora4}")
    print(f"DEBUG - LoRA multipliers: {lora1_multiplier}, {lora2_multiplier}, {lora3_multiplier}, {lora4_multiplier}")
    print(f"DEBUG - LoRA folder: {lora_folder}")
    
    # Convert values safely to float or None
    try:
        slg_start_float = float(slg_start) if slg_start is not None and str(slg_start).lower() != "none" else None
    except (ValueError, TypeError):
        slg_start_float = None
        print(f"Warning: Could not convert slg_start '{slg_start}' to float")
    
    try:
        slg_end_float = float(slg_end) if slg_end is not None and str(slg_end).lower() != "none" else None
    except (ValueError, TypeError):
        slg_end_float = None
        print(f"Warning: Could not convert slg_end '{slg_end}' to float")
    
    print(f"slg_start_float: {slg_start_float}, slg_end_float: {slg_end_float}")
    
    if stop_event.is_set():
        yield [], [], "", "" # Yield empty previews
        return

    run_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
    unique_preview_suffix = f"wanx_{run_id}" # Add prefix for clarity
    # --- Construct unique preview paths ---
    preview_base_path = os.path.join(save_path, f"latent_preview_{unique_preview_suffix}")
    preview_mp4_path = preview_base_path + ".mp4"
    preview_png_path = preview_base_path + ".png"

    # Check if this is a Fun-Control task
    is_fun_control = "-FC" in task and control_video is not None
    if is_fun_control:
        print(f"DEBUG - Using Fun-Control mode with control video: {control_video}")
        # Verify control video is provided
        if not control_video:
            yield [], "Error: No control video provided", "Fun-Control requires a control video"
            return
        
        # Verify needed files exist
        for path_name, path in [
            ("DIT", dit_path),
            ("VAE", vae_path),
            ("T5", t5_path),
            ("CLIP", clip_path)
        ]:
            if not os.path.exists(path):
                yield [], f"Error: {path_name} model not found", f"Model file doesn't exist: {path}"
                return
    
    # Get current seed or use provided seed
    current_seed = seed
    if seed == -1:
        current_seed = random.randint(0, 2**32 - 1)
        
    # Check if we need input image (required for i2v, not for t2v)
    if "i2v" in task and not input_image:
        yield [], "Error: No input image provided", "Please provide an input image for image-to-video generation"
        return
        
    # Check for Fun-Control requirements
    if is_fun_control and not control_video:
        yield [], "Error: No control video provided", "Please provide a control video for Fun-Control generation"
        return

    # Prepare environment
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    
    clear_cuda_cache()

    # Fix 2: Create command array with all string values
    command = [
        sys.executable,
        "wan_generate_video.py",
        "--task", str(task),
        "--prompt", str(prompt),
        "--video_size", str(height), str(width),
        "--video_length", str(video_length),
        "--fps", str(fps),
        "--infer_steps", str(infer_steps),
        "--save_path", str(save_path),
        "--seed", str(current_seed),
        "--flow_shift", str(flow_shift),
        "--guidance_scale", str(guidance_scale),
        "--output_type", str(output_type),
        "--attn_mode", str(attn_mode),
        "--blocks_to_swap", str(block_swap),
        "--dit", str(dit_path),
        "--vae", str(vae_path),
        "--t5", str(t5_path),
        "--sample_solver", str(sample_solver)
    ]
    
    # Fix 3: Only add boolean flags if they're True
    if enable_preview and preview_steps > 0:
        command.extend(["--preview", str(preview_steps)])
        # --- ADDED: Pass the unique suffix ---
        command.extend(["--preview_suffix", unique_preview_suffix])
        # --- End Pass Suffix ---
        print(f"DEBUG - Enabling preview every {preview_steps} steps with suffix {unique_preview_suffix}.")

    if enable_cfg_skip and cfg_skip_mode != "none":
        command.extend([
            "--cfg_skip_mode", str(cfg_skip_mode),
            "--cfg_apply_ratio", str(cfg_apply_ratio)
        ])
        
    if wanx_input_end and wanx_input_end != "none" and os.path.exists(str(wanx_input_end)):
        command.extend(["--end_image_path", str(wanx_input_end)])
        command.extend(["--trim_tail_frames", "3"])
        
    # Handle Fun-Control (control video path)
    if is_fun_control and control_video:
        command.extend(["--control_path", str(control_video)])
        command.extend(["--control_weight", str(control_strength)])
        command.extend(["--control_start", str(control_start)])
        command.extend(["--control_end", str(control_end)])

    # Handle SLG parameters
    if slg_layers and str(slg_layers).strip() and str(slg_layers).lower() != "none":
        try:
            # Make sure slg_layers is parsed as a list of integers
            slg_list = []
            for layer in str(slg_layers).split(","):
                layer = layer.strip()
                if layer.isdigit():  # Only add if it's a valid integer
                    slg_list.append(int(layer))
            if slg_list:  # Only add if we have valid layers
                command.extend(["--slg_layers", ",".join(map(str, slg_list))])

                # Only add slg_start and slg_end if we have valid slg_layers
                try:
                    if slg_start_float is not None and slg_start_float >= 0:
                        command.extend(["--slg_start", str(slg_start_float)])
                    if slg_end_float is not None and slg_end_float <= 1.0:
                        command.extend(["--slg_end", str(slg_end_float)])
                except ValueError as e:
                    print(f"Invalid SLG timing values: {str(e)}")
        except ValueError as e:
            print(f"Invalid SLG layers format: {slg_layers} - {str(e)}")

    
    # Add image path only for i2v task and if input image is provided
    if "i2v" in task and input_image:
        command.extend(["--image_path", str(input_image)])
        command.extend(["--clip", str(clip_path)])  # CLIP is needed for i2v and Fun-Control
    
    # Add video path for v2v task
    if "v2v" in task and input_image:
        command.extend(["--video_path", str(input_image)])
        # Add strength parameter for video-to-video
        if isinstance(guidance_scale, (int, float)) and guidance_scale > 0:
            command.extend(["--strength", str(guidance_scale)])
    
    if negative_prompt:
        command.extend(["--negative_prompt", str(negative_prompt)])
    
    # Add boolean flags correctly
    if fp8:
        command.append("--fp8")
    
    if fp8_scaled:
        command.append("--fp8_scaled")
    
    if fp8_t5:
        command.append("--fp8_t5")
        
    if exclude_single_blocks:
        command.append("--exclude_single_blocks")
    
    # Handle LoRA weights and multipliers
    lora_weights = [lora1, lora2, lora3, lora4]
    lora_multipliers = [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier]
    
    valid_loras = []
    for weight, mult in zip(lora_weights, lora_multipliers):
        # Skip None, empty, or "None" values
        if weight is None or not str(weight) or str(weight).lower() == "none":
            continue
        
        # Ensure weight is a string
        weight_str = str(weight)

        # Construct full path and verify file exists
        full_path = os.path.join(lora_folder, weight_str)
        if not os.path.exists(full_path):
            print(f"LoRA file not found: {full_path}")
            continue

        # Add valid LoRA to the list
        valid_loras.append((full_path, mult))

    # Only add LoRA parameters if we have valid LoRAs
    if valid_loras:
        weights = [w for w, _ in valid_loras]
        multipliers = [str(m) for _, m in valid_loras]
        command.extend(["--lora_weight"] + weights)
        command.extend(["--lora_multiplier"] + multipliers)
    
    # Make sure every item in command is a string
    command = [str(item) for item in command]
    
    print(f"Running: {' '.join(command)}")

    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    videos = []
    processed_preview_files = set() # Keep track of previews already yielded - REMAINS THE SAME IN UI FUNCTION
    # --- Reset preview state for this run ---
    current_preview_yield_path = None
    last_preview_mtime = 0
    
    current_phase = "Preparing" # Add phase tracking like FramePack
    while True:
        if stop_event.is_set():
            try:
                p.terminate()
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            except Exception as e:
                print(f"Error terminating subprocess: {e}")
            yield [], [], "Generation stopped by user.", "" # Yield empty previews
            return

        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            time.sleep(0.01); continue

        line = line.strip()
        if not line: continue
        print(f"WANX SUBPROCESS: {line}") # Log subprocess output

        # --- Adopt FramePack's Parsing Logic ---
        status_text = f"Processing (seed: {current_seed})" # Default status
        progress_text_update = line # Default progress

        # Check for TQDM progress using regex
        tqdm_match = re.search(r'(\d+)\%\|.+\| (\d+)/(\d+) \[(\d{2}:\d{2})<(\d{2}:\d{2})', line)

        if tqdm_match:
             percentage = int(tqdm_match.group(1))
             current_step = int(tqdm_match.group(2))
             total_steps = int(tqdm_match.group(3))
             time_elapsed = tqdm_match.group(4)
             time_remaining = tqdm_match.group(5)

             current_phase = f"Denoising Step {current_step}/{total_steps}" # Update phase

             # Format progress text like FramePack for JS compatibility
             progress_text_update = f"Step {current_step}/{total_steps} ({percentage}%) | Elapsed: {time_elapsed}, Remaining: {time_remaining}"
             status_text = f"Generating (seed: {current_seed}) - {current_phase}"

        elif "ERROR" in line.upper() or "TRACEBACK" in line.upper():
             status_text = f"Error (seed: {current_seed})"
             progress_text_update = line # Show error line
             current_phase = "Error"

        # Add more phases if needed (e.g., "Decoding", "Saving") by checking logs
        elif "Decoding video..." in line: # Placeholder check
             current_phase = "Decoding Video"
             status_text = f"Generating (seed: {current_seed}) - {current_phase}"
             progress_text_update = "Decoding video..."

        elif "Video saved to:" in line: # Placeholder check
             current_phase = "Saved"
             status_text = f"Completed (seed: {current_seed})"
             progress_text_update = line # Show the save line
        # Add any other status parsing if needed
        preview_updated = False
        current_mtime = 0
        found_preview_path = None

        if enable_preview:
            # --- MODIFIED: Check unique paths ---
            if os.path.exists(preview_mp4_path):
                current_mtime = os.path.getmtime(preview_mp4_path)
                found_preview_path = preview_mp4_path
            elif os.path.exists(preview_png_path):
                current_mtime = os.path.getmtime(preview_png_path)
                found_preview_path = preview_png_path
            # --- End Modified Check ---

            if found_preview_path and current_mtime > last_preview_mtime:
                print(f"DEBUG: Preview file updated: {found_preview_path} (mtime: {current_mtime})")
                # Yield the clean path (already unique)
                current_preview_yield_path = found_preview_path # No cache buster needed
                last_preview_mtime = current_mtime
                preview_updated = True
        # --- End Preview Check ---

        # --- YIELD ---
        # Yield progress and potentially updated unique preview path
            preview_list_for_yield = [current_preview_yield_path] if current_preview_yield_path else []
            # Yield progress and potentially updated unique preview path list
            yield videos.copy(), preview_list_for_yield, status_text, progress_text_update

    p.stdout.close()
    rc = p.wait() 

    clear_cuda_cache()
    time.sleep(0.5)

    # --- Collect final generated video ---
    generated_video_path = None
    if rc == 0: # Only look for video if process succeeded
        save_path_abs = os.path.abspath(save_path)
        if os.path.exists(save_path_abs):
            # Find the most recent mp4 containing the seed
            all_mp4_files = glob.glob(os.path.join(save_path_abs, f"*_{current_seed}*.mp4"))
            # Exclude files in the 'previews' subdirectory
            all_mp4_files = [f for f in all_mp4_files if "previews" not in os.path.dirname(f)]

            if all_mp4_files:
                # Find the *absolute* most recent one, as multiple might match seed in edge cases
                generated_video_path = max(all_mp4_files, key=os.path.getmtime)
                print(f"Found newly generated video: {generated_video_path}")

                # Add metadata (assuming add_metadata_to_video exists and works)
                parameters = {
                    "prompt": prompt, "negative_prompt": negative_prompt,
                    "input_image": input_image if "i2v" in task else None,
                    "width": width, "height": height, "video_length": video_length, "fps": fps,
                    "infer_steps": infer_steps, "flow_shift": flow_shift, "guidance_scale": guidance_scale,
                    "seed": current_seed, "task": task, "dit_path": dit_path,
                    "vae_path": vae_path, "t5_path": t5_path, "clip_path": clip_path if "i2v" in task or is_fun_control else None,
                    "save_path": save_path, "output_type": output_type, "sample_solver": sample_solver,
                    "exclude_single_blocks": exclude_single_blocks, "attn_mode": attn_mode,
                    "block_swap": block_swap, "fp8": fp8, "fp8_scaled": fp8_scaled, "fp8_t5": fp8_t5,
                    "lora_weights": [lora1, lora2, lora3, lora4],
                    "lora_multipliers": [lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier],
                    "slg_layers": slg_layers, "slg_start": slg_start, "slg_end": slg_end,
                    "enable_cfg_skip": enable_cfg_skip, "cfg_skip_mode": cfg_skip_mode, "cfg_apply_ratio": cfg_apply_ratio,
                    "control_video": control_video if is_fun_control else None,
                    "control_strength": control_strength if is_fun_control else None,
                    "control_start": control_start if is_fun_control else None,
                    "control_end": control_end if is_fun_control else None,
                }
                try:
                     add_metadata_to_video(generated_video_path, parameters)
                except NameError:
                     print("Warning: add_metadata_to_video function not found. Skipping metadata.")
                except Exception as meta_err:
                     print(f"Warning: Failed to add metadata: {meta_err}")

                # Append to the final video list
                videos.append((str(generated_video_path), f"Seed: {current_seed}"))
            else:
                 print(f"Subprocess finished successfully (rc=0), but could not find generated video for seed {current_seed} in {save_path_abs}")

# --- Final Yield ---
    final_status = f"Completed (seed: {current_seed})" if rc == 0 and generated_video_path else f"Failed (seed: {current_seed}, rc={rc})"
    final_progress = f"Video saved: {os.path.basename(generated_video_path)}" if rc == 0 and generated_video_path else f"Subprocess failed with exit code {rc}"

    # Check for the preview file one last time for the final update (using unique path)
    # --- MODIFIED Final Preview Check and List Creation ---
    final_preview_path = None
    # --- Use the UNIQUE paths defined earlier in the function ---
    if os.path.exists(preview_mp4_path):
        final_preview_path = os.path.abspath(preview_mp4_path)
    elif os.path.exists(preview_png_path):
        final_preview_path = os.path.abspath(preview_png_path)
    # --- End path checking ---

    final_preview_list_for_yield = [final_preview_path] if final_preview_path else []
    # --- End Modified ---

    yield videos, final_preview_list_for_yield, final_status, final_progress

def send_wanx_to_v2v(
    gallery: list,
    prompt: str,
    selected_index: int,
    width: int,
    height: int,
    video_length: int,
    fps: int,
    infer_steps: int,
    seed: int,
    flow_shift: float,
    guidance_scale: float,
    negative_prompt: str
) -> Tuple:
    """Send the selected WanX video to Video2Video tab"""
    if gallery is None or not gallery:
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)
    
    # If no selection made but we have videos, use the first one
    if selected_index is None and len(gallery) > 0:
        selected_index = 0
        
    if selected_index is None or selected_index >= len(gallery):
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)

    selected_item = gallery[selected_index]
    
    # Handle different gallery item formats
    if isinstance(selected_item, tuple):
        video_path = selected_item[0]
    elif isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    else:
        video_path = selected_item

    # Clean up path for Video component
    if isinstance(video_path, tuple):
        video_path = video_path[0]
        
    # Make sure it's a string
    video_path = str(video_path)

    return (video_path, prompt, width, height, video_length, fps, infer_steps, seed, 
            flow_shift, guidance_scale, negative_prompt)

def wanx_generate_video_batch(
    prompt,
    negative_prompt,
    width,
    height,
    video_length,
    fps,
    infer_steps,
    flow_shift,
    guidance_scale,
    seed,
    task,
    dit_path,
    vae_path,
    t5_path,
    clip_path,
    save_path,
    output_type,
    sample_solver,
    exclude_single_blocks,
    attn_mode,
    block_swap,
    fp8,
    fp8_scaled,
    fp8_t5,
    lora_folder,
    slg_layers: int,
    slg_start: Optional[str],
    slg_end: Optional[str],
    lora1="None",
    lora2="None",
    lora3="None",
    lora4="None",
    lora1_multiplier=1.0,
    lora2_multiplier=1.0,
    lora3_multiplier=1.0,
    lora4_multiplier=1.0,
    batch_size=1,
    input_image=None  # Make input_image optional and place it at the end
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    """Generate videos with WanX with support for batches"""
    slg_start = None if slg_start == 'None' or slg_start is None else slg_start
    slg_end = None if slg_end == 'None' or slg_end is None else slg_end

    # Now safely convert to float if not None
    slg_start_float = float(slg_start) if slg_start is not None and isinstance(slg_start, (str, int, float)) else None
    slg_end_float = float(slg_end) if slg_end is not None and isinstance(slg_end, (str, int, float)) else None
    print(f"slg_start_float: {slg_start_float}, slg_end_float: {slg_end_float}")
    global stop_event
    stop_event.clear()
    
    all_videos = []
    progress_text = "Starting generation..."
    yield [], "Preparing...", progress_text
    
    # Process each item in the batch
    for i in range(batch_size):
        if stop_event.is_set():
            yield all_videos, "Generation stopped by user", ""
            return
            
        # Calculate seed for this batch item
        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1:
            current_seed = seed + i
            
        batch_text = f"Generating video {i + 1} of {batch_size}"
        yield all_videos.copy(), batch_text, progress_text
        
        # Generate a single video using the existing function
        for videos, status, progress in wanx_generate_video(
            prompt, 
            negative_prompt, 
            input_image, 
            width, 
            height, 
            video_length, 
            fps, 
            infer_steps, 
            flow_shift, 
            guidance_scale, 
            current_seed,
            task, 
            dit_path, 
            vae_path, 
            t5_path, 
            clip_path, 
            save_path, 
            output_type, 
            sample_solver, 
            exclude_single_blocks,
            attn_mode, 
            block_swap, 
            fp8, 
            fp8_scaled, 
            fp8_t5,
            lora_folder,
            slg_layers,
            slg_start,
            slg_end,
            lora1,
            lora2,
            lora3,
            lora4,
            lora1_multiplier,
            lora2_multiplier,
            lora3_multiplier,
            lora4_multiplier
        ):
            if videos:
                all_videos.extend(videos)
            yield all_videos.copy(), f"Batch {i+1}/{batch_size}: {status}", progress
    
    yield all_videos, "Batch complete", ""

def update_wanx_t2v_dimensions(size):
    """Update width and height based on selected size"""
    width, height = map(int, size.split('*'))
    return gr.update(value=width), gr.update(value=height)

def handle_wanx_t2v_gallery_select(evt: gr.SelectData) -> int:
    """Track selected index when gallery item is clicked"""
    return evt.index

def send_wanx_t2v_to_v2v(
    gallery, prompt, selected_index, width, height, video_length,
    fps, infer_steps, seed, flow_shift, guidance_scale, negative_prompt
) -> Tuple:
    """Send the selected WanX T2V video to Video2Video tab"""
    if not gallery or selected_index is None or selected_index >= len(gallery):
        return (None, "", width, height, video_length, fps, infer_steps, seed, 
                flow_shift, guidance_scale, negative_prompt)

    selected_item = gallery[selected_index]

    if isinstance(selected_item, dict):
        video_path = selected_item.get("name", selected_item.get("data", None))
    elif isinstance(selected_item, (tuple, list)):
        video_path = selected_item[0]
    else:
        video_path = selected_item

    if isinstance(video_path, tuple):
        video_path = video_path[0]

    return (str(video_path), prompt, width, height, video_length, fps, infer_steps, seed, 
            flow_shift, guidance_scale, negative_prompt)

def prepare_for_batch_extension(input_img, base_video, batch_size):
    """Prepare inputs for batch video extension"""
    if input_img is None:
        return None, None, batch_size, "No input image found", ""
        
    if base_video is None:
        return input_img, None, batch_size, "No base video selected for extension", ""
        
    return input_img, base_video, batch_size, "Preparing batch extension...", f"Will create {batch_size} variations of extended video"

def concat_batch_videos(base_video_path, generated_videos, save_path, original_video_path=None):
    """Concatenate multiple generated videos with the base video"""
    if not base_video_path:
        return [], "No base video provided"
            
    if not generated_videos or len(generated_videos) == 0:
        return [], "No new videos generated"
    
    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Track all extended videos
    extended_videos = []
    
    # For each generated video, create an extended version
    for i, video_item in enumerate(generated_videos):
        try:
            # Extract video path from gallery item
            if isinstance(video_item, tuple):
                new_video_path = video_item[0]
                seed_info = video_item[1] if len(video_item) > 1 else ""
            elif isinstance(video_item, dict):
                new_video_path = video_item.get("name", video_item.get("data", None))
                seed_info = ""
            else:
                new_video_path = video_item
                seed_info = ""
                
            if not new_video_path or not os.path.exists(new_video_path):
                print(f"Skipping missing video: {new_video_path}")
                continue
                
            # Create unique output filename
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
            # Extract seed from seed_info if available
            seed_match = re.search(r"Seed: (\d+)", seed_info)
            seed_part = f"_seed{seed_match.group(1)}" if seed_match else f"_{i}"
            
            output_filename = f"extended_{timestamp}{seed_part}_{Path(base_video_path).stem}.mp4"
            output_path = os.path.join(save_path, output_filename)
            
            # Create a temporary file list for ffmpeg
            list_file = os.path.join(save_path, f"temp_list_{i}.txt")
            with open(list_file, "w") as f:
                f.write(f"file '{os.path.abspath(base_video_path)}'\n")
                f.write(f"file '{os.path.abspath(new_video_path)}'\n")
            
            # Run ffmpeg concatenation
            command = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            
            # Clean up temporary file
            if os.path.exists(list_file):
                os.remove(list_file)
                
            # Add to extended videos list if successful
            if os.path.exists(output_path):
                seed_display = f"Extended {seed_info}" if seed_info else f"Extended video #{i+1}"
                extended_videos.append((output_path, seed_display))
            
        except Exception as e:
            print(f"Error processing video {i}: {str(e)}")
    
    if not extended_videos:
        return [], "Failed to create any extended videos"
        
    return extended_videos, f"Successfully created {len(extended_videos)} extended videos"

def wanx_extend_single_video(
    prompt, negative_prompt, input_image, base_video_path,
    width, height, video_length, fps, infer_steps, 
    flow_shift, guidance_scale, seed, 
    task, dit_path, vae_path, t5_path, clip_path,
    save_path, output_type, sample_solver, exclude_single_blocks,
    attn_mode, block_swap, fp8, fp8_scaled, fp8_t5, lora_folder,
    slg_layers="", slg_start=0.0, slg_end=1.0,
    lora1="None", lora2="None", lora3="None", lora4="None",
    lora1_multiplier=1.0, lora2_multiplier=1.0, lora3_multiplier=1.0, lora4_multiplier=1.0
):
    """Generate a single video and concatenate with base video"""
    # First, generate the video with proper parameter handling
    all_videos = []
    
    # Sanitize lora parameters
    lora_weights = [str(lora1) if lora1 is not None else "None", 
                   str(lora2) if lora2 is not None else "None", 
                   str(lora3) if lora3 is not None else "None", 
                   str(lora4) if lora4 is not None else "None"]
    
    # Convert multipliers to float
    try:
        lora_multipliers = [float(lora1_multiplier), float(lora2_multiplier), 
                           float(lora3_multiplier), float(lora4_multiplier)]
    except (ValueError, TypeError):
        # Fallback to defaults if conversion fails
        lora_multipliers = [1.0, 1.0, 1.0, 1.0]
    
    # Debug print
    print(f"Sanitized LoRA weights: {lora_weights}")
    print(f"Sanitized LoRA multipliers: {lora_multipliers}")
    
    # Generate video
    for videos, status, progress in wanx_generate_video(
        prompt, negative_prompt, input_image, width, height, 
        video_length, fps, infer_steps, flow_shift, guidance_scale, 
        seed, task, dit_path, vae_path, t5_path, clip_path, 
        save_path, output_type, sample_solver, exclude_single_blocks,
        attn_mode, block_swap, fp8, fp8_scaled, fp8_t5, lora_folder,
        slg_layers, slg_start, slg_end,
        lora_weights[0], lora_weights[1], lora_weights[2], lora_weights[3],
        lora_multipliers[0], lora_multipliers[1], lora_multipliers[2], lora_multipliers[3],
        enable_cfg_skip=False,
        cfg_skip_mode="none",
        cfg_apply_ratio=0.7
    ):
        
        # Keep track of generated videos
        if videos:
            all_videos = videos
        
        # Forward progress updates
        yield all_videos, status, progress
    
    # Now concatenate with base video if we have something
    if all_videos and base_video_path and os.path.exists(base_video_path):
        try:
            print(f"Extending base video: {base_video_path}")
            
            # Create unique output filename
            timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
            output_filename = f"extended_{timestamp}_seed{seed}_{Path(base_video_path).stem}.mp4"
            output_path = os.path.join(save_path, output_filename)
            
            # Extract the path from the gallery item
            new_video_path = all_videos[0][0] if isinstance(all_videos[0], tuple) else all_videos[0]
            
            # Create a temporary file list for ffmpeg
            list_file = os.path.join(save_path, f"temp_list_{seed}.txt")
            with open(list_file, "w") as f:
                f.write(f"file '{os.path.abspath(base_video_path)}'\n")
                f.write(f"file '{os.path.abspath(new_video_path)}'\n")
            
            print(f"Concatenating: {base_video_path} + {new_video_path}")
            
            # Run ffmpeg concatenation
            command = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                "-y",
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            
            # Clean up temporary file
            if os.path.exists(list_file):
                os.remove(list_file)
                
            # Return the extended video if successful
            if os.path.exists(output_path):
                extended_video = [(output_path, f"Extended (Seed: {seed})")]
                print(f"Successfully created extended video: {output_path}")
                yield extended_video, "Extended video created successfully", ""
                return
            else:
                print(f"Failed to create extended video at {output_path}")
        except Exception as e:
            print(f"Error creating extended video: {str(e)}")
    
    # If we got here, something went wrong with the concatenation
    yield all_videos, "Generated video (extension failed)", ""

def process_batch_extension(
    prompt, negative_prompt, input_image, base_video,
    width, height, video_length, fps, infer_steps,
    flow_shift, guidance_scale, seed, batch_size,
    task, dit_folder, dit_path, vae_path, t5_path, clip_path, # <<< Added dit_folder
    save_path, output_type, sample_solver, exclude_single_blocks,
    attn_mode, block_swap, fp8, fp8_scaled, fp8_t5, lora_folder,
    slg_layers, slg_start, slg_end,
    lora1="None", lora2="None", lora3="None", lora4="None",
    lora1_multiplier=1.0, lora2_multiplier=1.0, lora3_multiplier=1.0, lora4_multiplier=1.0
):
    """Process a batch of video extensions one at a time"""
    global stop_event
    stop_event.clear()

    all_extended_videos = [] # Store successfully extended videos
    progress_text = "Starting video extension batch..."
    yield [], progress_text, "" # Initial yield

    try:
        # Ensure batch_size is treated as an integer
        batch_size = int(batch_size)
    except (ValueError, TypeError):
        batch_size = 1
        print("Warning: Invalid batch_size, defaulting to 1.")

    # Ensure base_video exists
    if not base_video or not os.path.exists(base_video):
        yield [], "Error: Base video not found", f"Cannot find video at {base_video}"
        return

    # Process each batch item independently
    for i in range(batch_size):
        if stop_event.is_set():
            yield all_extended_videos, "Extension stopped by user", ""
            return

        # Calculate seed for this batch item
        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif batch_size > 1:
            current_seed = seed + i

        batch_text = f"Processing extension {i+1}/{batch_size} (seed: {current_seed})"
        yield all_extended_videos, batch_text, progress_text # Update progress

        # Use the direct wrapper with correct parameter order, including dit_folder
        generation_iterator = wanx_extend_video_wrapper(
            prompt=prompt, negative_prompt=negative_prompt, input_image=input_image, base_video_path=base_video,
            width=width, height=height, video_length=video_length, fps=fps, infer_steps=infer_steps,
            flow_shift=flow_shift, guidance_scale=guidance_scale, seed=current_seed,
            task=task,
            dit_folder=dit_folder, # <<< Pass the folder path
            dit_path=dit_path,     # <<< Pass the model filename
            vae_path=vae_path,
            t5_path=t5_path,
            clip_path=clip_path,
            save_path=save_path, output_type=output_type, sample_solver=sample_solver,
            exclude_single_blocks=exclude_single_blocks, attn_mode=attn_mode, block_swap=block_swap,
            fp8=fp8, fp8_scaled=fp8_scaled, fp8_t5=fp8_t5, lora_folder=lora_folder,
            slg_layers=slg_layers, slg_start=slg_start, slg_end=slg_end,
            lora1=lora1, lora2=lora2, lora3=lora3, lora4=lora4,
            lora1_multiplier=lora1_multiplier, lora2_multiplier=lora2_multiplier,
            lora3_multiplier=lora3_multiplier, lora4_multiplier=lora4_multiplier
        )

        # Iterate through the generator for this single extension
        final_videos_for_item = []
        final_status_for_item = "Unknown status"
        final_progress_for_item = ""
        try:
            for videos, status, progress in generation_iterator:
                # Forward progress information immediately
                yield all_extended_videos, f"Batch {i+1}/{batch_size}: {status}", progress

                # Store the latest state for this item
                final_videos_for_item = videos
                final_status_for_item = status
                final_progress_for_item = progress

            # After the loop for one item finishes, check the result
            if final_videos_for_item:
                 # Check if the video is actually an extended one
                is_extended = any("Extended" in (v[1] if isinstance(v, tuple) else "") for v in final_videos_for_item)
                if is_extended:
                    all_extended_videos.extend(final_videos_for_item)
                    print(f"Added extended video to collection (total: {len(all_extended_videos)})")
                else:
                    # It was just the generated segment, maybe log this?
                    print(f"Video segment generated for batch {i+1} but extension failed or wasn't performed.")
            else:
                print(f"No video returned for batch item {i+1}.")


        except Exception as e:
            print(f"Error during single extension processing (batch {i+1}): {e}")
            yield all_extended_videos, f"Error in batch {i+1}: {e}", ""


        # Clean CUDA cache between generations
        clear_cuda_cache()
        time.sleep(0.5)

    # Final yield after the loop
    yield all_extended_videos, "Batch extension complete", ""

def handle_extend_generation(base_video_path: str, new_videos: list, save_path: str, current_gallery: list) -> tuple:
    """Combine generated video with base video and update gallery"""
    if not base_video_path:
        return current_gallery, "Extend failed: No base video provided"
        
    if not new_videos:
        return current_gallery, "Extend failed: No new video generated"
    
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Get the first video from new_videos (gallery item)
    new_video_path = new_videos[0][0] if isinstance(new_videos[0], tuple) else new_videos[0]
    
    # Create a unique output filename
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    output_filename = f"extended_{timestamp}_{Path(base_video_path).stem}.mp4"
    output_path = str(Path(save_path) / output_filename)
    
    try:
        # Concatenate the videos using ffmpeg
        (
            ffmpeg
            .input(base_video_path)
            .concat(
                ffmpeg.input(new_video_path)
            )
            .output(output_path)
            .run(overwrite_output=True, quiet=True)
        )
        
        # Create a new gallery entry with the combined video
        updated_gallery = [(output_path, f"Extended video: {Path(output_path).stem}")]
        
        return updated_gallery, f"Successfully extended video to {Path(output_path).name}"
    except Exception as e:
        print(f"Error extending video: {str(e)}")
        return current_gallery, f"Failed to extend video: {str(e)}"

# UI setup
with gr.Blocks(
    theme=themes.Default(
        primary_hue=colors.Color(
            name="custom",
            c50="#E6F0FF",
            c100="#CCE0FF",
            c200="#99C1FF",
            c300="#66A3FF",
            c400="#3384FF",
            c500="#0060df",  # This is your main color
            c600="#0052C2",
            c700="#003D91",
            c800="#002961",
            c900="#001430",
            c950="#000A18"
        )
    ),
    css="""
    .gallery-item:first-child { border: 2px solid #4CAF50 !important; }
    .gallery-item:first-child:hover { border-color: #45a049 !important; }
    .green-btn {
        background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
        color: white !important;
        border: none !important;
    }
    .green-btn:hover {
        background: linear-gradient(to bottom right, #27ae60, #219651) !important;
    }
    .refresh-btn {
        max-width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    """, 

) as demo:
    # Add state for tracking selected video indices in both tabs
    selected_index = gr.State(value=None)  # For Text to Video
    v2v_selected_index = gr.State(value=None)  # For Video to Video
    params_state = gr.State() #New addition
    i2v_selected_index = gr.State(value=None) 
    skyreels_selected_index = gr.State(value=None)
    wanx_i2v_selected_index = gr.State(value=None)
    extended_videos = gr.State(value=[])
    wanx_base_video = gr.State(value=None)
    wanx_sharpest_frame_number = gr.State(value=None)  
    wanx_sharpest_frame_path = gr.State(value=None)   
    wanx_trimmed_video_path = gr.State(value=None) 
    wanx_v2v_selected_index = gr.State(value=None)
    wanx_t2v_selected_index = gr.State(value=None)
    framepack_selected_index = gr.State(value=None)
    framepack_original_dims = gr.State(value="")
    demo.load(None, None, None, js="""
    () => {
        document.title = 'H1111';

        function updateTitle(text) {
            if (text && text.trim()) {
                // Regex for the FramePack format: "Item ... (...)% | ... Remaining: HH:MM"
                const framepackMatch = text.match(/.*?\((\d+)%\).*?Remaining:\s*(\d{2}:\d{2})/);
                // Regex for standard tqdm format (like WanX uses)
                const tqdmMatch = text.match(/(\d+)%\|.*\[.*<(\d{2}:\d{2})/); // Adjusted slightly for robustness

                if (framepackMatch) {
                    // Handle FramePack format
                    const percentage = framepackMatch[1];
                    const timeRemaining = framepackMatch[2];
                    document.title = `[${percentage}% ETA: ${timeRemaining}] - H1111`;
                } else if (tqdmMatch) { // <<< ADDED ELSE IF for standard tqdm
                    // Handle standard tqdm format
                    const percentage = tqdmMatch[1];
                    const timeRemaining = tqdmMatch[2];
                    document.title = `[${percentage}% ETA: ${timeRemaining}] - H1111`;
                } else {
                    // Optional: Reset title if neither format matches?
                    // document.title = 'H1111';
                }
            }
        }

        setTimeout(() => {
            // This selector should still find all relevant progress textareas
            const progressElements = document.querySelectorAll('textarea.scroll-hide');
            progressElements.forEach(element => {
                if (element) {
                    new MutationObserver(() => {
                        updateTitle(element.value);
                    }).observe(element, {
                        attributes: true,
                        childList: true,
                        characterData: true
                    });
                }
            });
        }, 1000);
    }
    """)
        
    with gr.Tabs() as tabs:

        #FRAME PACK TAB
        with gr.Tab(id=10, label="FramePack") as framepack_tab:
            
            with gr.Row():
                with gr.Column(scale=4):
                    framepack_prompt = gr.Textbox(
                        scale=3, label="Prompt (Supports sections: index:prompt;;;index:prompt)",
                        value="cinematic video of a cat wizard casting a spell", lines=3,
                        info="Use '0:prompt;;;-1:prompt' or '0-2:prompt;;;3:prompt'. Index total sections -1 is last section."
                    )
                    framepack_negative_prompt = gr.Textbox(scale=3, label="Negative Prompt", value="", lines=3)
                with gr.Column(scale=1):
                    framepack_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    framepack_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
                    framepack_is_f1 = gr.Checkbox(label="🏎️ Use F1 Model", value=False,
                                                  info="Switches to the F1 model (different DiT path and logic).")                    
                with gr.Column(scale=2):
                    framepack_batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                    framepack_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")
            with gr.Row():
                framepack_generate_btn = gr.Button("Generate FramePack Video", elem_classes="green-btn")
                framepack_stop_btn = gr.Button("Stop Generation", variant="stop")

            # Main Content
            with gr.Row():
                # --- Left Column ---
                with gr.Column():
                    framepack_input_image = gr.Image(label="Input Image (Video Start)", type="filepath")
                    with gr.Accordion("Optional End Frame Control", open=False):
                        framepack_input_end_frame = gr.Image(label="End Frame Image (Video End)", type="filepath", scale=1)
                        framepack_end_frame_influence = gr.Dropdown(
                            label="End Frame Influence Mode",
                            choices=["last", "half", "progressive", "bookend"],
                            value="last",
                            info="How the end frame affects generation (if provided)",
                            visible=False
                        )
                        framepack_end_frame_weight = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.5, # Default changed from 0.3
                            label="End Frame Weight",
                            info="Influence strength of the end frame (if provided)",
                            visible=False
                        )
                    with gr.Accordion("Optional Control Video", open=False):
                        framepack_input_video = gr.Video(label="Input Video (Overrides Start Image for V2V History)", format="mp4", type="filepath", interactive=True, info="Used for V2V if provided.")

                    gr.Markdown("### Resolution Options (Choose One)")
                    framepack_target_resolution = gr.Number(
                        label="Option 1: Target Resolution (Uses Buckets)",
                        value=640, minimum=128, maximum=1280, step=32,
                        info="Target bucket size (e.g., 640 for 640x640). Uses input image aspect ratio. Final size divisible by 32.",
                        interactive=True
                    )
                    with gr.Accordion("Option 2: Explicit Resolution (Overrides Option 1)", open=False):
                         framepack_scale_slider = gr.Slider(
                             minimum=1, maximum=200, value=100, step=1, label="Scale % (UI Only)"
                         )
                         with gr.Row():
                             framepack_width = gr.Number(
                                 label="Width", value=None, minimum=64, step=32, 
                                 info="Must be divisible by 32.", interactive=True
                             )
                             framepack_calc_height_btn = gr.Button("→")
                             framepack_calc_width_btn = gr.Button("←")
                             framepack_height = gr.Number(
                                 label="Height", value=None, minimum=64, step=32,
                                 info="Must be divisible by 32.", interactive=True
                             )
                    framepack_total_second_length = gr.Slider(minimum=1.0, maximum=120.0, step=0.5, label="Total Video Length (seconds)", value=5.0)
                    framepack_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Output FPS", value=30)
                    with gr.Row():
                        framepack_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        framepack_random_seed =gr.Button("🎲️")
                    framepack_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Steps", value=25, interactive=True) # Moved here

                # --- Right Column ---
                with gr.Column():
                    framepack_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2], rows=[1],
                        object_fit="contain", height="auto", show_label=True,
                        elem_id="gallery_framepack", allow_preview=True, preview=True
                    )
                    with gr.Accordion("Latent Preview (During Generation)", open=True):
                        with gr.Row():
                            framepack_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                            framepack_preview_every_n_sections = gr.Slider(
                                minimum=1, maximum=50, step=1, value=1,
                                label="Preview Every N Sections",
                                info="Generates previews during the sampling loop."
                            )
                        framepack_preview_output = gr.Video( # Changed from Gallery to Video
                             label="Latest Preview", height=300,
                             interactive=False, # Not interactive for display
                             elem_id="framepack_preview_video"
                        )
                    with gr.Group():
                        with gr.Row():
                            framepack_refresh_lora_btn = gr.Button("🔄 LoRA", elem_classes="refresh-btn") # Specific LoRA refresh
                            framepack_lora_folder = gr.Textbox(label="LoRa Folder", value="lora", scale=4)
                        framepack_lora_weights = []
                        framepack_lora_multipliers = []
                        for i in range(4): # Assuming max 4 LoRAs like other tabs
                            with gr.Row():
                                framepack_lora_weights.append(gr.Dropdown(
                                    label=f"LoRA {i+1}", choices=get_lora_options("lora"),
                                    value="None", allow_custom_value=False, interactive=True, scale=2
                                ))
                                framepack_lora_multipliers.append(gr.Slider(
                                    label=f"Multiplier", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1, interactive=True
                                ))
                    # Fixed Generation Parameters Section
                    with gr.Accordion("Generation Parameters", open=True):
                        with gr.Row():
                            framepack_distilled_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Distilled Guidance Scale (embedded_cfg_scale)", value=10.0, interactive=True)
                            framepack_guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, label="Guidance Scale (CFG)", value=1.0, interactive=True, info="Default 1.0 (no CFG), backend recommends not changing.")
                        with gr.Row():
                            framepack_guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="CFG Rescale (rs)", value=0.0, interactive=True, info="Default 0.0, backend recommends not changing.")
                            framepack_latent_window_size = gr.Number(label="Latent Window Size", value=9, interactive=True, info="Default 9")
                            framepack_sample_solver = gr.Dropdown(label="Sample Solver", choices=["unipc", "dpm++", "vanilla"], value="unipc", interactive=True)

            with gr.Accordion("Advanced Section Control (Optional)", open=False):
                gr.Markdown(
                    "Define specific prompts and starting images for different sections of the video. "
                    "For the index you can input a range or a single index. A 5 second default video has 4 sections. The first section is 0 and the last is 3"
                )
                # --- Define section controls explicitly ---
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**--- Control Slot 1 ---**")
                            with gr.Row():
                                 
                                framepack_sec_1 = gr.Textbox(label="Index/Range", value="0", placeholder="e.g., 0 or 0-1", interactive=True)
                            framepack_sec_prompt_1 = gr.Textbox(label="Prompt Override", lines=2, placeholder="Overrides base prompt for these sections")
                            framepack_sec_image_1 = gr.Image(label="Start Image Override", type="filepath", scale=1)
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**--- Control Slot 2 ---**")
                            with gr.Row():
                                 
                                framepack_sec_2 = gr.Textbox(label="Index/Range", value="1", placeholder="e.g., 2 or 2-3", interactive=True)
                            framepack_sec_prompt_2 = gr.Textbox(label="Prompt Override", lines=2)
                            framepack_sec_image_2 = gr.Image(label="Start Image Override", type="filepath", scale=1)
                with gr.Row():
                    with gr.Column(scale=1):
                         with gr.Group():
                            gr.Markdown("**--- Control Slot 3 ---**")
                            with gr.Row():
                                
                                framepack_sec_3 = gr.Textbox(label="Index/Range", value="2", placeholder="e.g., 4 or 4-5", interactive=True)
                            framepack_sec_prompt_3 = gr.Textbox(label="Prompt Override", lines=2)
                            framepack_sec_image_3 = gr.Image(label="Start Image Override", type="filepath", scale=1)
                    with gr.Column(scale=1):
                         with gr.Group():
                            gr.Markdown("**--- Control Slot 4 ---**")
                            with gr.Row():
                                framepack_sec_4 = gr.Textbox(label="Index/Range", value="3", placeholder="e.g., 6 or 6-7", interactive=True)
                            framepack_sec_prompt_4 = gr.Textbox(label="Prompt Override", lines=2)
                            framepack_sec_image_4 = gr.Image(label="Start Image Override", type="filepath", scale=1)

                # Group section control components for easier passing to functions (remains the same)
                framepack_secs = [framepack_sec_1, framepack_sec_2, framepack_sec_3, framepack_sec_4]
                framepack_sec_prompts = [framepack_sec_prompt_1, framepack_sec_prompt_2, framepack_sec_prompt_3, framepack_sec_prompt_4]
                framepack_sec_images = [framepack_sec_image_1, framepack_sec_image_2, framepack_sec_image_3, framepack_sec_image_4]                            

            # Performance/Memory Accordion - Updated
            with gr.Accordion("Performance / Memory", open=True):
                with gr.Row():
                    framepack_fp8 = gr.Checkbox(label="Use FP8 DiT", value=False, info="Enable FP8 precision for the main Transformer model.")
                    framepack_fp8_llm = gr.Checkbox(label="Use FP8 LLM (Text Encoder 1)", value=False, info="Enable FP8 for the Llama text encoder.")
                    framepack_fp8_scaled = gr.Checkbox(label="Use Scaled FP8 DiT", value=False, info="Requires FP8 DiT. Use scaled math (potential quality improvement).")
                    framepack_blocks_to_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Blocks to Swap (to Save VRAM, 0=disable)", value=26,
                                    info="Higher values = less VRAM usage but slower generation")
                    framepack_bulk_decode = gr.Checkbox(label="Bulk Decode Frames (Faster Decode, Higher VRAM)", value=False, info="Decode all frames at once instead of section by section.")
                with gr.Row():
                    framepack_attn_mode = gr.Dropdown(
                        label="Attention Mode",
                        choices=["torch", "sdpa", "flash", "xformers", "sageattn"], # Added choices from script
                        value="sdpa", # Defaulting to sdpa
                        interactive=True
                    )
                    framepack_vae_chunk_size = gr.Number(label="VAE Chunk Size (CausalConv3d)", value=32, step=1, minimum=0, info="0 or None=disable (Default: None)")
                    framepack_vae_spatial_tile_sample_min_size = gr.Number(label="VAE Spatial Tile Min Size", value=128, step=16, minimum=0, info="0 or None=disable (Default: None)")
                    framepack_device = gr.Textbox(label="Device Override (optional)", placeholder="e.g., cuda:0, cpu")
                with gr.Row():
                    framepack_use_teacache = gr.Checkbox(label="Use TeaCache", value=False, info="Enable TeaCache for faster generation (shits hands).")
                    framepack_teacache_steps = gr.Number(label="TeaCache Init Steps", value=25, step=1, minimum=1, info="Steps for TeaCache init (match Inference Steps)")
                    framepack_teacache_thresh = gr.Slider(label="TeaCache Threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.15, info="Relative L1 distance threshold for skipping.")                    

            with gr.Accordion("Model Paths / Advanced", open=False):
                 with gr.Row():
                    framepack_transformer_path = gr.Textbox(label="Transformer Path (DiT)", value="hunyuan/FramePackI2V_HY_bf16.safetensors", interactive=True)
                    framepack_vae_path = gr.Textbox(label="VAE Path", value="hunyuan/pytorch_model.pt")
                 with gr.Row():
                    framepack_text_encoder_path = gr.Textbox(label="Text Encoder 1 (Llama) Path *Required*", value="hunyuan/llava_llama3_fp16.safetensors")
                    framepack_text_encoder_2_path = gr.Textbox(label="Text Encoder 2 (CLIP) Path *Required*", value="hunyuan/clip_l.safetensors")
                 with gr.Row():
                    framepack_image_encoder_path = gr.Textbox(label="Image Encoder (SigLIP) Path *Required*", value="hunyuan/model.safetensors")
                    framepack_save_path = gr.Textbox(label="Save Path *Required*", value="outputs")
        # Text to Video Tab
        with gr.Tab(id=1, label="Hunyuan-t2v"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt = gr.Textbox(scale=3, label="Enter your prompt", value="POV video of a cat chasing a frob.", lines=5)

                with gr.Column(scale=1):
                    token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    
                    t2v_width = gr.Slider(minimum=64, maximum=1536, step=16, value=544, label="Video Width")
                    t2v_height = gr.Slider(minimum=64, maximum=1536, step=16, value=544, label="Video Height")
                    video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=25, elem_id="my_special_slider")
                    fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24, elem_id="my_special_slider")
                    infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=30, elem_id="my_special_slider")
                    flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=11.0, elem_id="my_special_slider")
                    cfg_scale = gr.Slider(minimum=0.0, maximum=14.0, step=0.1, label="cfg Scale", value=7.0, elem_id="my_special_slider")
            
                with gr.Column():

                    with gr.Row():
                        video_output = gr.Gallery(
                            label="Generated Videos (Click to select)",
                            columns=[2],
                            rows=[2],
                            object_fit="contain",
                            height="auto",
                            show_label=True,
                            elem_id="gallery",
                            allow_preview=True,
                            preview=True
                        )
                    with gr.Row():send_t2v_to_v2v_btn = gr.Button("Send Selected to Video2Video")
            
            with gr.Row():
                    refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    lora_weights = []
                    lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))            
            with gr.Row():
                exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")
                model = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("hunyuan"),
                    value="mp_rank_00_model_states.pt",
                    allow_custom_value=True,
                    interactive=True
                )
                vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                save_path = gr.Textbox(label="Save Path", value="outputs")
            with gr.Row():
                lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                use_fp8 = gr.Checkbox(label="Use FP8 (faster but lower precision)", value=True)
                attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)

        #Image to Video Tab
        with gr.Tab(label="Hunyuan-i2v") as i2v_tab: # Keep tab name consistent if needed elsewhere
            # ... (Keep existing Rows for prompt, batch size, progress) ...
            with gr.Row():
                with gr.Column(scale=4):
                    i2v_prompt = gr.Textbox(scale=3, label="Enter your prompt", value="POV video of a cat chasing a frob.", lines=5)

                with gr.Column(scale=1):
                    i2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    i2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    i2v_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress_i2v") # Unique elem_id
                    i2v_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text_i2v") # Unique elem_id

            with gr.Row():
                i2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                i2v_stop_btn = gr.Button("Stop Generation", variant="stop")


            with gr.Row():
                with gr.Column():
                    i2v_input = gr.Image(label="Input Image", type="filepath")
                    # REMOVED i2v_strength slider, as hv_i2v_generate_video.py doesn't seem to use it based on the sample command
                    # i2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength")
                    scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale % (UI Only - affects W/H)") # Clarified UI only
                    original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)
                    # Width and height inputs
                    with gr.Row():
                        # Renamed width/height to avoid potential conflicts if they weren't already prefixed
                        i2v_width = gr.Number(label="New Width", value=720, step=16) # Default from sample
                        calc_height_btn = gr.Button("→")
                        calc_width_btn = gr.Button("←")
                        i2v_height = gr.Number(label="New Height", value=720, step=16) # Default from sample
                    i2v_video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=49) # Default from sample
                    i2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24) # Default from sample
                    i2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=30) # Default from sample
                    i2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=17.0) # Default from sample
                    i2v_cfg_scale = gr.Slider(minimum=0.0, maximum=14.0, step=0.1, label="Embedded CFG Scale", value=7.0) # Default from sample
                    i2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale (CFG)", value=1.0) # Default from sample (usually 1.0 for no CFG)

                with gr.Column():
                    i2v_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery_i2v", # Unique elem_id
                        allow_preview=True,
                        preview=True
                    )
                    i2v_send_to_v2v_btn = gr.Button("Send Selected to Hunyuan-v2v") # Keep sending to original V2V

                    # Add LoRA section for Image2Video
                    i2v_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    i2v_lora_weights = []
                    i2v_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            i2v_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}",
                                choices=get_lora_options(),
                                value="None",
                                allow_custom_value=True,
                                interactive=True
                            ))
                            i2v_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier",
                                minimum=0.0,
                                maximum=2.0,
                                step=0.05,
                                value=1.0
                            ))

            with gr.Row():
                i2v_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                i2v_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                i2v_dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")
                i2v_model = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("hunyuan"),
                    value="mp_rank_00_model_states_i2v.pt", # Default from sample
                    allow_custom_value=True,
                    interactive=True
                )
                i2v_vae = gr.Textbox(label="VAE Path", value="hunyuan/pytorch_model.pt") # Default from sample
                i2v_te1 = gr.Textbox(label="Text Encoder 1 Path", value="hunyuan/llava_llama3_fp16.safetensors") # Default from sample
                i2v_te2 = gr.Textbox(label="Text Encoder 2 Path", value="hunyuan/clip_l.safetensors") # Default from sample
                i2v_clip_vision_path = gr.Textbox(label="CLIP Vision Path", value="hunyuan/llava_llama3_vision.safetensors") # Default from sample
                i2v_save_path = gr.Textbox(label="Save Path", value="outputs") # Default from sample
            with gr.Row():
                i2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                i2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video") # Default from sample
                i2v_use_split_attn = gr.Checkbox(label="Use Split Attention", value=False) # Not in sample, keep default False
                i2v_use_fp8 = gr.Checkbox(label="Use FP8 DiT", value=False) # Not in sample, keep default False
                i2v_fp8_llm = gr.Checkbox(label="Use FP8 LLM", value=False) # Not in sample, keep default False
                i2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa") # Default from sample
                i2v_block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=30) # Default from sample
                # Add VAE tiling options like sample command
                i2v_vae_chunk_size = gr.Number(label="VAE Chunk Size", value=32, step=1, info="For CausalConv3d, set 0 to disable")
                i2v_vae_spatial_tile_min = gr.Number(label="VAE Spatial Tile Min Size", value=128, step=16, info="Set 0 to disable spatial tiling")

        # Video to Video Tab
        with gr.Tab(id=2, label="Hunyuan v2v") as v2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    v2v_prompt = gr.Textbox(scale=3, label="Enter your prompt", value="POV video of a cat chasing a frob.", lines=5)
                    v2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt (for SkyReels models)",
                        value="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
                        lines=3
                    )

                with gr.Column(scale=1):
                    v2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    v2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    v2v_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    v2v_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                v2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                v2v_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    v2v_input = gr.Video(label="Input Video", format="mp4")
                    v2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength")
                    v2v_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    v2v_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)

                    # Width and Height Inputs
                    with gr.Row():
                        v2v_width = gr.Number(label="New Width", value=544, step=16)
                        v2v_calc_height_btn = gr.Button("→")
                        v2v_calc_width_btn = gr.Button("←")
                        v2v_height = gr.Number(label="New Height", value=544, step=16)
                    v2v_video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=25)
                    v2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24)
                    v2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=30)
                    v2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=11.0)
                    v2v_cfg_scale = gr.Slider(minimum=0.0, maximum=14.0, step=0.1, label="cfg scale", value=7.0)
                with gr.Column():
                    v2v_output = gr.Gallery(
                        label="Generated Videos",
                        columns=[1],
                        rows=[1],
                        object_fit="contain",
                        height="auto"
                    )
                    v2v_send_to_input_btn = gr.Button("Send Selected to Input")  # New button
                    v2v_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    v2v_lora_weights = []
                    v2v_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            v2v_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            v2v_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))

            with gr.Row():
                v2v_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)                
                v2v_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                v2v_dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")
                v2v_model = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("hunyuan"),
                    value="mp_rank_00_model_states.pt",
                    allow_custom_value=True,
                    interactive=True
                )
                v2v_vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                v2v_te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                v2v_te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                v2v_save_path = gr.Textbox(label="Save Path", value="outputs")
            with gr.Row():
                v2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                v2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                v2v_use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                v2v_use_fp8 = gr.Checkbox(label="Use FP8 (faster but lower precision)", value=True)
                v2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                v2v_block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)
                v2v_split_uncond = gr.Checkbox(label="Split Unconditional (for SkyReels)", value=True)

### SKYREELS

        with gr.Tab(label="SkyReels-i2v") as skyreels_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    skyreels_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A person walking on a beach at sunset", 
                        lines=5
                    )
                    skyreels_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
                        lines=3
                    )

                with gr.Column(scale=1):
                    skyreels_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    skyreels_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    skyreels_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    skyreels_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                skyreels_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                skyreels_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    skyreels_input = gr.Image(label="Input Image (optional)", type="filepath")
                    with gr.Row():
                        skyreels_use_random_folder = gr.Checkbox(label="Use Random Images from Folder", value=False)
                        skyreels_input_folder = gr.Textbox(
                            label="Image Folder Path", 
                            placeholder="Path to folder containing images",
                            visible=False
                        )
                        skyreels_folder_status = gr.Textbox(
                            label="Folder Status", 
                            placeholder="Status will appear here",
                            interactive=False,
                            visible=False
                        )
                        skyreels_validate_folder_btn = gr.Button("Validate Folder", visible=False)
                    skyreels_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength")

                    # Scale slider as percentage 
                    skyreels_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    skyreels_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)

                    # Width and height inputs
                    with gr.Row():
                        skyreels_width = gr.Number(label="New Width", value=544, step=16)
                        skyreels_calc_height_btn = gr.Button("→")
                        skyreels_calc_width_btn = gr.Button("←")
                        skyreels_height = gr.Number(label="New Height", value=544, step=16)

                    skyreels_video_length = gr.Slider(minimum=1, maximum=201, step=1, label="Video Length in Frames", value=25)
                    skyreels_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=24)
                    skyreels_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=30)
                    skyreels_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=11.0)
                    skyreels_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=6.0)
                    skyreels_embedded_cfg_scale = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, label="Embedded CFG Scale", value=1.0)

                with gr.Column():
                    skyreels_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery",
                        allow_preview=True,
                        preview=True
                    )
                    skyreels_send_to_v2v_btn = gr.Button("Send Selected to Video2Video")

                    # Add LoRA section for SKYREELS
                    skyreels_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    skyreels_lora_weights = []
                    skyreels_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            skyreels_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            skyreels_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))
            with gr.Row():
                skyreels_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)                
                skyreels_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                skyreels_dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")
                skyreels_model = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("skyreels"),
                    value="skyreels_hunyuan_i2v_bf16.safetensors",
                    allow_custom_value=True,
                    interactive=True
                )
                skyreels_vae = gr.Textbox(label="vae", value="hunyuan/pytorch_model.pt")
                skyreels_te1 = gr.Textbox(label="te1", value="hunyuan/llava_llama3_fp16.safetensors")
                skyreels_te2 = gr.Textbox(label="te2", value="hunyuan/clip_l.safetensors")
                skyreels_save_path = gr.Textbox(label="Save Path", value="outputs")

            with gr.Row():
                skyreels_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                skyreels_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                skyreels_use_split_attn = gr.Checkbox(label="Use Split Attention", value=False)
                skyreels_use_fp8 = gr.Checkbox(label="Use FP8 (faster but lower precision)", value=True)
                skyreels_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                skyreels_block_swap = gr.Slider(minimum=0, maximum=36, step=1, label="Block Swap to Save Vram", value=0)
                skyreels_split_uncond = gr.Checkbox(label="Split Unconditional", value=True)

        # WanX Image to Video Tab
        with gr.Tab(id=4, label="WanX-i2v") as wanx_i2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    wanx_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A person walking on a beach at sunset", 
                        lines=5
                    )
                    wanx_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        lines=3,
                    )

                with gr.Column(scale=1):
                    wanx_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    wanx_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    wanx_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    wanx_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                wanx_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                wanx_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    wanx_input = gr.Image(label="Input Image", type="filepath")
                    with gr.Row():
                        wanx_use_random_folder = gr.Checkbox(label="Use Random Images from Folder", value=False)
                        wanx_input_folder = gr.Textbox(
                            label="Image Folder Path", 
                            placeholder="Path to folder containing images",
                            visible=False
                        )
                        wanx_folder_status = gr.Textbox(
                            label="Folder Status", 
                            placeholder="Status will appear here",
                            interactive=False,
                            visible=False
                        )
                        wanx_validate_folder_btn = gr.Button("Validate Folder", visible=False)
                    with gr.Row():
                        wanx_use_end_image = gr.Checkbox(label="use ending image", value=False)
                        wanx_input_end = gr.Image(label="End Image", type="filepath", visible=False)
                        wanx_trim_frames = gr.Checkbox(label="trim last 3 frames", value=True, visible=False, interactive=True)

                    with gr.Row():
                        wanx_use_fun_control = gr.Checkbox(label="Use Fun-Control Model", value=False)
                        wanx_control_video = gr.Video(label="Control Video for Fun-Control", visible=False, format="mp4")
                        wanx_control_strength = gr.Slider(minimum=0.1, maximum=2.0, step=0.05, value=1.0, 
                            label="Control Strength", visible=False,
                            info="Adjust influence of control video (1.0 = normal)")
                        wanx_control_start = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.0,
                            label="Control Start (Fun-Control fade-in)",
                            visible=False,
                            info="When (0-1) in the timeline control influence is full after fade-in"
                        )
                        wanx_control_end = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=1.0,
                            label="Control End (Fun-Control fade-out start)",
                            visible=False,
                            info="When (0-1) in the timeline control starts to fade out"
                        )
                    wanx_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    wanx_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)
        
                    # Width and height display
                    with gr.Row():
                        wanx_width = gr.Number(label="Width", value=832, interactive=True)
                        wanx_calc_height_btn = gr.Button("→")
                        wanx_calc_width_btn = gr.Button("←")
                        wanx_height = gr.Number(label="Height", value=480, interactive=True)
                        wanx_recommend_flow_btn = gr.Button("Recommend Flow Shift", size="sm")

                    wanx_video_length = gr.Slider(minimum=1, maximum=401, step=4, label="Video Length in Frames", value=81)
                    wanx_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    wanx_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=20)
                    wanx_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=3.0, 
                                            info="Recommended: 3.0 for 480p, 5.0 for others")
                    wanx_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, label="Guidance Scale", value=5.0)

                with gr.Column():
                    wanx_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery",
                        allow_preview=True,
                        preview=True
                    )
                    with gr.Accordion("Latent Preview (During Generation)", open=True):
                        wanx_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                        wanx_preview_steps = gr.Slider(minimum=1, maximum=50, step=1, value=5,
                                                       label="Preview Every N Steps", info="Generates previews during the sampling loop.")
                        wanx_preview_output = gr.Gallery(
                            label="Latent Previews", columns=4, rows=2, object_fit="contain", height=300,
                            allow_preview=True, preview=True, show_label=True, elem_id="wanx_preview_gallery"
                        )                    
                    wanx_send_to_v2v_btn = gr.Button("Send Selected to Hunyuan-v2v")
                    wanx_i2v_send_to_wanx_v2v_btn = gr.Button("Send Selected to WanX-v2v")
                    wanx_send_last_frame_btn = gr.Button("Send Last Frame to Input")
                    wanx_extend_btn = gr.Button("Extend Video")
                    wanx_frames_to_check = gr.Slider(minimum=1, maximum=100, step=1, value=30, 
                                                   label="Frames to Check from End", 
                                                   info="Number of frames from the end to check for sharpness")
                    wanx_send_sharpest_frame_btn = gr.Button("Extract Sharpest Frame")
                    wanx_trim_and_extend_btn = gr.Button("Trim Video & Prepare for Extension")
                    wanx_sharpest_frame_status = gr.Textbox(label="Status", interactive=False)

                # Add a new button for directly extending with the trimmed video
                    wanx_extend_with_trimmed_btn = gr.Button("Extend with Trimmed Video")

                    # Add LoRA section for WanX-i2v similar to other tabs
                    wanx_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    wanx_lora_weights = []
                    wanx_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            wanx_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            wanx_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))

            with gr.Row():
                wanx_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                # Update the wanx_task dropdown choices to include Fun-Control options
                wanx_task = gr.Dropdown(
                    label="Task",
                    choices=["i2v-14B", "i2v-14B-FC", "i2v-14B-FC-1.1", "t2v-14B", "t2v-1.3B", "t2v-14B-FC", "t2v-1.3B-FC", "i2v-1.3B-new"],
                    value="i2v-14B",
                    info="Select model type. *-FC options enable Fun-Control features"
                )
                wanx_dit_folder = gr.Textbox(label="DiT Model Folder", value="wan")
                wanx_dit_path = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("wan"),  # Use the existing function to get available models
                    value="wan2.1_i2v_720p_14B_fp16.safetensors",
                    allow_custom_value=True,
                    interactive=True
                )
                wanx_vae_path = gr.Textbox(label="VAE Path", value="wan/Wan2.1_VAE.pth")
                wanx_t5_path = gr.Textbox(label="T5 Path", value="wan/models_t5_umt5-xxl-enc-bf16.pth")
                wanx_clip_path = gr.Textbox(label="CLIP Path", value="wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
                wanx_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                wanx_save_path = gr.Textbox(label="Save Path", value="outputs")

            with gr.Row():
                wanx_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                wanx_sample_solver = gr.Radio(choices=["unipc", "dpm++", "vanilla"], label="Sample Solver", value="unipc")
                wanx_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                wanx_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                wanx_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=0)
                
                with gr.Column():
                    wanx_fp8 = gr.Checkbox(label="Use FP8", value=True)
                    wanx_fp8_scaled = gr.Checkbox(label="Use Scaled FP8", value=False, info="For mixing fp16/bf16 and fp8 weights")
                    wanx_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)

            # Add new row for Skip Layer Guidance options
            with gr.Row():
                wanx_slg_layers = gr.Textbox(label="SLG Layers", value="", placeholder="Comma-separated layer indices, e.g. 1,5,10", info="Layers to skip for guidance")
                wanx_slg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG Start", value=0.0, info="When to start skipping layers (% of total steps)")
                wanx_slg_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG End", value=1.0, info="When to stop skipping layers (% of total steps)")    
            
            with gr.Row():
                wanx_enable_cfg_skip = gr.Checkbox(label="Enable CFG Skip (similar to teacache)", value=False)
                with gr.Column(visible=False) as wanx_cfg_skip_options:
                    wanx_cfg_skip_mode = gr.Radio(
                        choices=["early", "late", "middle", "early_late", "alternate", "none"],
                        label="CFG Skip Mode",
                        value="none",
                        info="Controls which steps to apply CFG on"
                    )
                    wanx_cfg_apply_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                        label="CFG Apply Ratio", 
                        info="Ratio of steps to apply CFG (0.0-1.0). Lower values = faster, but less accurate"
                    )

        #WanX-t2v Tab

        # WanX Text to Video Tab
        with gr.Tab(id=5, label="WanX-t2v") as wanx_t2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    wanx_t2v_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A person walking on a beach at sunset", 
                        lines=5
                    )
                    wanx_t2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="",
                        lines=3,
                        info="Leave empty to use default negative prompt"
                    )

                with gr.Column(scale=1):
                    wanx_t2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    wanx_t2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    wanx_t2v_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    wanx_t2v_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                wanx_t2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                wanx_t2v_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        wanx_t2v_width = gr.Number(label="Width", value=832, interactive=True, info="Should be divisible by 32")
                        wanx_t2v_height = gr.Number(label="Height", value=480, interactive=True, info="Should be divisible by 32")
                        wanx_t2v_recommend_flow_btn = gr.Button("Recommend Flow Shift", size="sm")

                    wanx_t2v_video_length = gr.Slider(minimum=1, maximum=201, step=4, label="Video Length in Frames", value=81)
                    wanx_t2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    wanx_t2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=20)
                    wanx_t2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=5.0, 
                                             info="Recommended: 3.0 for I2V with 480p, 5.0 for others")
                    wanx_t2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=5.0)

                with gr.Column():
                    wanx_t2v_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery",
                        allow_preview=True,
                        preview=True
                    )
                    with gr.Accordion("Latent Preview (During Generation)", open=False):
                        wanx_t2v_enable_preview = gr.Checkbox(label="Enable Latent Preview", value=False)
                        wanx_t2v_preview_steps = gr.Slider(minimum=1, maximum=50, step=1, value=5,
                                                        label="Preview Every N Steps", info="Generates previews during the sampling loop.")
                        wanx_t2v_preview_output = gr.Gallery(
                            label="Latent Previews", columns=4, rows=2, object_fit="contain", height=300,
                            allow_preview=True, preview=True, show_label=True, elem_id="wanx_t2v_preview_gallery"
                        )                    
                    wanx_t2v_send_to_v2v_btn = gr.Button("Send Selected to Hunyuan v2v")
                    wanx_t2v_send_to_wanx_v2v_btn = gr.Button("Send Selected to WanX-v2v")

                    # Add LoRA section for WanX-t2v
                    wanx_t2v_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    wanx_t2v_lora_weights = []
                    wanx_t2v_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            wanx_t2v_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            wanx_t2v_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))

            with gr.Row():
                wanx_t2v_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                wanx_t2v_task = gr.Dropdown(
                    label="Task",
                    choices=["t2v-1.3B", "t2v-14B", "t2i-14B"],
                    value="t2v-14B",
                    info="Select model size: t2v-1.3B is faster, t2v-14B has higher quality"
                )
                wanx_t2v_dit_path = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("wan"),
                    value="wan2.1_t2v_14B_fp16.safetensors",
                    allow_custom_value=True,
                    interactive=True
                )
                wanx_t2v_vae_path = gr.Textbox(label="VAE Path", value="wan/Wan2.1_VAE.pth")
                wanx_t2v_t5_path = gr.Textbox(label="T5 Path", value="wan/models_t5_umt5-xxl-enc-bf16.pth")
                wanx_t2v_clip_path = gr.Textbox(label="CLIP Path", visible=False, value="")
                wanx_t2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                wanx_t2v_save_path = gr.Textbox(label="Save Path", value="outputs")

            with gr.Row():
                wanx_t2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                wanx_t2v_sample_solver = gr.Radio(choices=["unipc", "dpm++", "vanilla"], label="Sample Solver", value="unipc")
                wanx_t2v_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                wanx_t2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                wanx_t2v_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=0, 
                                         info="Max 39 for 14B model, 29 for 1.3B model")
                
                with gr.Column():
                    wanx_t2v_fp8 = gr.Checkbox(label="Use FP8", value=True)
                    wanx_t2v_fp8_scaled = gr.Checkbox(label="Use Scaled FP8", value=False,
                                                info="For mixing fp16/bf16 and fp8 weights")
                    wanx_t2v_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)
            
            # Add new row for Skip Layer Guidance options
            with gr.Row():
                wanx_t2v_slg_layers = gr.Textbox(label="SLG Layers", value="", placeholder="Comma-separated layer indices, e.g. 1,5,10", info="Layers to skip for guidance")
                wanx_t2v_slg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG Start", value=0.0, info="When to start skipping layers (% of total steps)")
                wanx_t2v_slg_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG End", value=1.0, info="When to stop skipping layers (% of total steps)")
                wanx_t2v_use_random_folder = gr.Checkbox(visible=False, value=False, label="Use Random Images")
                wanx_t2v_input_folder = gr.Textbox(visible=False, value="", label="Image Folder")
                wanx_t2v_input_end = gr.Textbox(visible=False, value="none", label="End Frame")
            
            with gr.Row():
                wanx_t2v_enable_cfg_skip = gr.Checkbox(label="Enable CFG Skip (similar to teacache)", value=False)
                with gr.Column(visible=False) as wanx_t2v_cfg_skip_options:
                    wanx_t2v_cfg_skip_mode = gr.Radio(
                        choices=["early", "late", "middle", "early_late", "alternate", "none"],
                        label="CFG Skip Mode",
                        value="none",
                        info="Controls which steps to apply CFG on"
                    )
                    wanx_t2v_cfg_apply_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                        label="CFG Apply Ratio", 
                        info="Ratio of steps to apply CFG (0.0-1.0). Lower values = faster, but less accurate"
                    )

        #WanX-v2v Tab
        with gr.Tab(id=6, label="WanX-v2v") as wanx_v2v_tab:
            with gr.Row():
                with gr.Column(scale=4):
                    wanx_v2v_prompt = gr.Textbox(
                        scale=3, 
                        label="Enter your prompt", 
                        value="A person walking on a beach at sunset", 
                        lines=5
                    )
                    wanx_v2v_negative_prompt = gr.Textbox(
                        scale=3,
                        label="Negative Prompt",
                        value="",
                        lines=3,
                        info="Leave empty to use default negative prompt"
                    )

                with gr.Column(scale=1):
                    wanx_v2v_token_counter = gr.Number(label="Prompt Token Count", value=0, interactive=False)
                    wanx_v2v_batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)

                with gr.Column(scale=2):
                    wanx_v2v_batch_progress = gr.Textbox(label="", visible=True, elem_id="batch_progress")
                    wanx_v2v_progress_text = gr.Textbox(label="", visible=True, elem_id="progress_text")

            with gr.Row():
                wanx_v2v_generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
                wanx_v2v_stop_btn = gr.Button("Stop Generation", variant="stop")

            with gr.Row():
                with gr.Column():
                    wanx_v2v_input = gr.Video(label="Input Video", format="mp4")
                    wanx_v2v_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.75, label="Denoise Strength", 
                                                info="0 = keep original, 1 = full generation")
                    wanx_v2v_scale_slider = gr.Slider(minimum=1, maximum=200, value=100, step=1, label="Scale %")
                    wanx_v2v_original_dims = gr.Textbox(label="Original Dimensions", interactive=False, visible=True)

                    # Width and Height Inputs
                    with gr.Row():
                        wanx_v2v_width = gr.Number(label="New Width", value=832, step=32)
                        wanx_v2v_calc_height_btn = gr.Button("→")
                        wanx_v2v_calc_width_btn = gr.Button("←")
                        wanx_v2v_height = gr.Number(label="New Height", value=480, step=32)
                        wanx_v2v_recommend_flow_btn = gr.Button("Recommend Flow Shift", size="sm")

                    wanx_v2v_video_length = gr.Slider(minimum=1, maximum=201, step=4, label="Video Length in Frames", value=81)
                    wanx_v2v_fps = gr.Slider(minimum=1, maximum=60, step=1, label="Frames Per Second", value=16)
                    wanx_v2v_infer_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Inference Steps", value=40)
                    wanx_v2v_flow_shift = gr.Slider(minimum=0.0, maximum=28.0, step=0.5, label="Flow Shift", value=5.0,
                                               info="Recommended: 3.0 for 480p, 5.0 for others")
                    wanx_v2v_guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Scale", value=5.0)

                with gr.Column():
                    wanx_v2v_output = gr.Gallery(
                        label="Generated Videos (Click to select)",
                        columns=[2],
                        rows=[2],
                        object_fit="contain",
                        height="auto",
                        show_label=True,
                        elem_id="gallery",
                        allow_preview=True,
                        preview=True
                    )
                    wanx_v2v_send_to_v2v_btn = gr.Button("Send Selected to Hunyuan-v2v")

                    # Add LoRA section for WanX-v2v
                    wanx_v2v_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
                    wanx_v2v_lora_weights = []
                    wanx_v2v_lora_multipliers = []
                    for i in range(4):
                        with gr.Column():
                            wanx_v2v_lora_weights.append(gr.Dropdown(
                                label=f"LoRA {i+1}", 
                                choices=get_lora_options(), 
                                value="None", 
                                allow_custom_value=True,
                                interactive=True
                            ))
                            wanx_v2v_lora_multipliers.append(gr.Slider(
                                label=f"Multiplier", 
                                minimum=0.0, 
                                maximum=2.0, 
                                step=0.05, 
                                value=1.0
                            ))

            with gr.Row():
                wanx_v2v_seed = gr.Number(label="Seed (use -1 for random)", value=-1)
                wanx_v2v_task = gr.Dropdown(
                    label="Task",
                    choices=["t2v-14B", "t2v-1.3B"],
                    value="t2v-14B",
                    info="Model size: t2v-1.3B is faster, t2v-14B has higher quality"
                )
                wanx_v2v_dit_folder = gr.Textbox(label="DiT Model Folder", value="wan")
                wanx_v2v_dit_path = gr.Dropdown(
                    label="DiT Model",
                    choices=get_dit_models("wan"),
                    value="wan2.1_t2v_14B_fp16.safetensors",
                    allow_custom_value=True,
                    interactive=True
                )
                wanx_v2v_vae_path = gr.Textbox(label="VAE Path", value="wan/Wan2.1_VAE.pth")
                wanx_v2v_t5_path = gr.Textbox(label="T5 Path", value="wan/models_t5_umt5-xxl-enc-bf16.pth")
                wanx_v2v_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                wanx_v2v_save_path = gr.Textbox(label="Save Path", value="outputs")

            with gr.Row():
                wanx_v2v_output_type = gr.Radio(choices=["video", "images", "latent", "both"], label="Output Type", value="video")
                wanx_v2v_sample_solver = gr.Radio(choices=["unipc", "dpm++", "vanilla"], label="Sample Solver", value="unipc")
                wanx_v2v_exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                wanx_v2v_attn_mode = gr.Radio(choices=["sdpa", "flash", "sageattn", "xformers", "torch"], label="Attention Mode", value="sdpa")
                wanx_v2v_block_swap = gr.Slider(minimum=0, maximum=39, step=1, label="Block Swap to Save VRAM", value=0,
                                           info="Max 39 for 14B model, 29 for 1.3B model")

                with gr.Column():
                    wanx_v2v_fp8 = gr.Checkbox(label="Use FP8", value=True)
                    wanx_v2v_fp8_scaled = gr.Checkbox(label="Use Scaled FP8", value=False,
                                                  info="For mixing fp16/bf16 and fp8 weights")
                    wanx_v2v_fp8_t5 = gr.Checkbox(label="Use FP8 for T5", value=False)

            # Add Skip Layer Guidance options
            with gr.Row():
                wanx_v2v_slg_layers = gr.Textbox(label="SLG Layers", value="", placeholder="Comma-separated layer indices, e.g. 1,5,10", info="Layers to skip for guidance")
                wanx_v2v_slg_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG Start", value=0.0, info="When to start skipping layers (% of total steps)")
                wanx_v2v_slg_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SLG End", value=1.0, info="When to stop skipping layers (% of total steps)")

            with gr.Row():
                wanx_v2v_enable_cfg_skip = gr.Checkbox(label="Enable CFG Skip (similar to teacache)", value=False)
                with gr.Column(visible=False) as wanx_v2v_cfg_skip_options:
                    wanx_v2v_cfg_skip_mode = gr.Radio(
                        choices=["early", "late", "middle", "early_late", "alternate", "none"],
                        label="CFG Skip Mode",
                        value="none",
                        info="Controls which steps to apply CFG on"
                    )
                    wanx_v2v_cfg_apply_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                        label="CFG Apply Ratio", 
                        info="Ratio of steps to apply CFG (0.0-1.0). Lower values = faster, but less accurate"
                    )

        #Video Info Tab
        with gr.Tab("Video Info") as video_info_tab:
            with gr.Row():
                video_input = gr.Video(label="Upload Video", interactive=True)
                metadata_output = gr.JSON(label="Generation Parameters")

            with gr.Row():
                send_to_framepack_btn = gr.Button("Send to FramePack", variant="primary")
                send_to_t2v_btn = gr.Button("Send to Text2Video", variant="primary")
                send_to_v2v_btn = gr.Button("Send to Video2Video", variant="primary")
                send_to_wanx_i2v_btn = gr.Button("Send to WanX-i2v", variant="primary")
                send_to_wanx_t2v_btn = gr.Button("Send to WanX-t2v", variant="primary")
                send_to_wanx_v2v_btn = gr.Button("Send to WanX-v2v", variant="primary")


            with gr.Row():
                status = gr.Textbox(label="Status", interactive=False)

        #Merge Model's tab        
        with gr.Tab("Convert LoRA") as convert_lora_tab:
            def suggest_output_name(file_obj) -> str:
                """Generate suggested output name from input file"""
                if not file_obj:
                    return ""
                # Get input filename without extension and add MUSUBI
                base_name = os.path.splitext(os.path.basename(file_obj.name))[0]
                return f"{base_name}_MUSUBI"

            def convert_lora(input_file, output_name: str, target_format: str) -> str:
                """Convert LoRA file to specified format"""
                try:
                    if input_file is None:
                        return "Error: No input file selected"

                    # Ensure output directory exists
                    os.makedirs("lora", exist_ok=True)

                    # Construct output path
                    output_path = os.path.join("lora", f"{output_name}.safetensors")

                    # Determine which script to use based on target_format
                    if target_format == "Hunyuan to FramePack":
                        script_name = "convert_hunyuan_to_framepack.py"
                        cmd = [
                            sys.executable,
                            script_name,
                            "--input", input_file.name,
                            "--output", output_path
                        ]
                        print(f"Using '{script_name}' to convert {input_file.name} to {output_path} for FramePack.")
                    else: # Existing logic for "default" and "other"
                        script_name = "convert_lora.py"
                        cmd = [
                            sys.executable,
                            script_name,
                            "--input", input_file.name,
                            "--output", output_path,
                            "--target", target_format.lower()
                        ]

                    print(f"Running conversion command: {' '.join(cmd)}")

                    # Check if the selected script file exists
                    if not os.path.exists(script_name):
                         return f"Error: Conversion script '{script_name}' not found. Please ensure it's in the same directory as h1111.py."

                    # Execute conversion
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    if result.returncode != 0:
                        return f"Conversion failed with exit code {result.returncode}. See console for details:\n{result.stderr}\n{result.stdout}"


                    if os.path.exists(output_path):
                        return f"Successfully converted LoRA to {output_path}"
                    else:
                        return "Error: Output file not created"

                except subprocess.CalledProcessError as e:
                    print(f"Subprocess error:\n{e.stderr}\n{e.stdout}")
                    return f"Error during conversion: {e.stderr}"
                except FileNotFoundError as e:
                    # This catches errors like python executable not found, or the script itself not found
                    print(f"File not found error: {e}")
                    return f"Error: Required file not found. Make sure Python and the conversion script are accessible. Details: {e}"
                except Exception as e:
                    traceback.print_exc() # Print traceback for debugging
                    return f"Error: {str(e)}"


            with gr.Row():
                input_file = gr.File(label="Input LoRA File", file_types=[".safetensors"])
                output_name = gr.Textbox(label="Output Name", placeholder="Output filename (without extension)")
                format_radio = gr.Radio(
                    choices=["default", "other", "Hunyuan to FramePack"], # <-- Added new choice here
                    value="default",
                    label="Target Format",
                    info="Choose 'default' for H1111/MUSUBI format, 'other' for diffusion pipe format, or 'Hunyuan to FramePack' for FramePack compatibility."
                )

            with gr.Row():
                convert_btn = gr.Button("Convert LoRA", variant="primary")
                status_output = gr.Textbox(label="Status", interactive=False)

            # Automatically update output name when file is selected
            input_file.change(
                fn=suggest_output_name,
                inputs=[input_file],
                outputs=[output_name]
            )

            # Handle conversion
            convert_btn.click(
                fn=convert_lora,
                inputs=[input_file, output_name, format_radio],
                outputs=status_output
            )
        with gr.Tab("Model Merging") as model_merge_tab:
            with gr.Row():
                with gr.Column():
                    # Model selection
                    dit_model = gr.Dropdown(
                        label="Base DiT Model",
                        choices=["mp_rank_00_model_states.pt"],
                        value="mp_rank_00_model_states.pt",
                        allow_custom_value=True,
                        interactive=True
                    )
                    merge_refresh_btn = gr.Button("🔄", elem_classes="refresh-btn")
            with gr.Row():
                with gr.Column():
                    # Output model name
                    output_model = gr.Textbox(label="Output Model Name", value="merged_model.safetensors")
                    exclude_single_blocks = gr.Checkbox(label="Exclude Single Blocks", value=False)
                    merge_btn = gr.Button("Merge Models", variant="primary")
                    merge_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                # LoRA selection section (similar to Text2Video)
                merge_lora_weights = []
                merge_lora_multipliers = []
                for i in range(4):
                    with gr.Column():
                        merge_lora_weights.append(gr.Dropdown(
                            label=f"LoRA {i+1}",
                            choices=get_lora_options(),
                            value="None",
                            allow_custom_value=True,
                            interactive=True
                        ))
                        merge_lora_multipliers.append(gr.Slider(
                            label=f"Multiplier",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.05,
                            value=1.0
                        ))
                with gr.Row():
                    merge_lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                    dit_folder = gr.Textbox(label="DiT Model Folder", value="hunyuan")

    #Event handlers etc

    # <<< Function to switch to FramePack Tab >>>
    def change_to_framepack_tab():
        return gr.Tabs(selected=10) # FramePack tab has id=10

    def handle_send_to_framepack_tab(metadata: dict) -> Tuple[str, dict, str]: # Added str return type for state value
        """Prepare parameters specifically for the FramePack tab."""
        if not metadata:
            # Return default/empty values for status, params, and original_dims state
            return "No parameters to send", {}, ""

        # Extract the value intended for the state here
        original_dims_value = metadata.get("original_dims_str", "")

        # Return status message, the full metadata for params_state, and the specific value for framepack_original_dims state
        return "Parameters ready for FramePack", metadata, original_dims_value

    send_to_framepack_btn.click(
            fn=handle_send_to_framepack_tab,
            inputs=[metadata_output],
            outputs=[status, params_state, framepack_original_dims] # Add framepack_original_dims here
        ).then(
            # This lambda now prepares updates for UI components (32 items)
            lambda params: (
                # Prepare the full list of 32 update values first
                (
                    # Fetch LoRA lists from params, default to empty lists if not found
                    (weights_from_meta := params.get("lora_weights", [])),
                    (mults_from_meta := params.get("lora_multipliers", [])),
                    # Create explicitly padded lists ensuring 4 elements
                    (padded_weights := (weights_from_meta + ["None"] * 4)[:4]),
                    (padded_mults := ([float(m) for m in mults_from_meta] + [1.0] * 4)[:4]), # Ensure multipliers are floats
    
                    # Build the list of update values
                    [
                        params.get("prompt", "cinematic video of a cat wizard casting a spell"),
                        params.get("negative_prompt", ""),
                        # Handle resolution: Prioritize explicit W/H if valid (divisible by 8), else use target_res, else default
                        gr_update(value=int(params["video_width"])) if params.get("video_width") and int(params.get("video_width", 0)) > 0 and int(params.get("video_width", 0)) % 8 == 0 else gr_update(value=None),
                        gr_update(value=int(params["video_height"])) if params.get("video_height") and int(params.get("video_height", 0)) > 0 and int(params.get("video_height", 0)) % 8 == 0 else gr_update(value=None),
                        # Use target resolution only if explicit width/height are *not* validly provided from metadata
                        gr_update(value=int(params.get("target_resolution"))) if not (params.get("video_width") and int(params.get("video_width", 0)) > 0 and int(params.get("video_width", 0)) % 8 == 0) and params.get("target_resolution") else gr_update(value=640),
                        params.get("video_seconds", 5.0),
                        params.get("fps", 30),
                        params.get("seed", -1),
                        params.get("infer_steps", 25),
                        params.get("embedded_cfg_scale", 10.0), # Distilled Guidance
                        params.get("guidance_scale", 1.0),      # CFG
                        params.get("guidance_rescale", 0.0),    # RS
                        params.get("sample_solver", "unipc"),
                        # Unpack the *padded* lists
                        *padded_weights, # 4 items
                        *padded_mults,   # 4 items
                        # Performance/Memory
                        params.get("fp8", False),
                        params.get("fp8_scaled", False),
                        params.get("fp8_llm", False),
                        params.get("blocks_to_swap", 26),
                        params.get("bulk_decode", False),
                        params.get("attn_mode", "sdpa"),
                        params.get("vae_chunk_size", 32),
                        params.get("vae_spatial_tile_sample_min_size", 128),
                        params.get("device", ""),
                        # End Frame Blending Params - Use UI defaults
                        params.get("end_frame_influence", "last"),
                        params.get("end_frame_weight", 0.5),
                        params.get("is_f1", False)
                    ]
                )[-1] # Return the list of values we just built
            ) if params else [gr.update()] * 32, 
            inputs=params_state, # Read parameters from state
            outputs=[
                # Map to FramePack components (UI only - 32 components)
                framepack_prompt,
                framepack_negative_prompt,
                framepack_width, # Will be updated or set to None
                framepack_height, # Will be updated or set to None
                framepack_target_resolution, # Will be updated or set to None/default
                framepack_total_second_length,
                framepack_fps,
                framepack_seed,
                framepack_steps,
                framepack_distilled_guidance_scale,
                framepack_guidance_scale,
                framepack_guidance_rescale,
                framepack_sample_solver,
                # LoRAs (unpacking the lists - 8 components total)
                *framepack_lora_weights, # 4 components
                *framepack_lora_multipliers, # 4 components
                 # Performance/Memory
                framepack_fp8,
                framepack_fp8_scaled,
                framepack_fp8_llm,
                framepack_blocks_to_swap,
                framepack_bulk_decode,
                framepack_attn_mode,
                framepack_vae_chunk_size,
                framepack_vae_spatial_tile_sample_min_size,
                framepack_device,
                # Map to new UI components
                framepack_end_frame_influence,
                framepack_end_frame_weight,
                framepack_is_f1
            ]
        ).then(
            fn=change_to_framepack_tab, # Switch to the FramePack tab
            inputs=None,
            outputs=[tabs]
        )
    # Connect FramePack Generate button
    def update_framepack_image_dimensions(image):
        """Update FramePack dimensions from uploaded image, store raw dims, set default target res"""
        if image is None:
            return "", gr.update(value=None), gr.update(value=None), gr.update(value=640) # Reset W/H, default target res
        try:
            img = Image.open(image)
            w, h = img.size
            original_dims_str = f"{w}x{h}" # Store raw WxH
            target_res_default = 640
            # Return original dims string, clear explicit W/H, set default target res
            return original_dims_str, gr.update(value=None), gr.update(value=None), gr.update(value=target_res_default)
        except Exception as e:
            print(f"Error reading image dimensions: {e}")
            return "", gr.update(value=None), gr.update(value=None), gr.update(value=640) # Fallback
        
    framepack_input_image.change(
        fn=update_framepack_image_dimensions,
        inputs=[framepack_input_image],
        outputs=[framepack_original_dims, framepack_width, framepack_height, framepack_target_resolution]
    )
        
    framepack_prompt.change(fn=count_prompt_tokens, inputs=framepack_prompt, outputs=framepack_token_counter)
    # If explicit width/height is set (and valid), clear target resolution
    def clear_target_res_on_explicit_change(val):
        return gr.update(value=None) if val is not None and val > 0 else gr.update()

    framepack_scale_slider.change(
        fn=update_framepack_from_scale,
        inputs=[framepack_scale_slider, framepack_original_dims],
        outputs=[framepack_width, framepack_height, framepack_target_resolution] # Also clears target res
    )

    framepack_calc_width_btn.click(
        fn=calculate_framepack_width,
        inputs=[framepack_height, framepack_original_dims],
        outputs=[framepack_width]
    ).then(
        fn=clear_target_res_on_explicit_change, # Clear target res if width is manually set
        inputs=[framepack_width],
        outputs=[framepack_target_resolution]
    )

    framepack_calc_height_btn.click(
        fn=calculate_framepack_height,
        inputs=[framepack_width, framepack_original_dims],
        outputs=[framepack_height]
    ).then(
        fn=clear_target_res_on_explicit_change, # Clear target res if height is manually set
        inputs=[framepack_height],
        outputs=[framepack_target_resolution]
    )

    framepack_width.change(
         fn=clear_target_res_on_explicit_change,
         inputs=[framepack_width],
         outputs=[framepack_target_resolution]
    )
    framepack_height.change(
         fn=clear_target_res_on_explicit_change,
         inputs=[framepack_height],
         outputs=[framepack_target_resolution]
    )

    # If target resolution is set (and valid), clear explicit width/height
    def clear_explicit_res_on_target_change(target_res):
        return (gr.update(value=None), gr.update(value=None)) if target_res is not None and target_res > 0 else (gr.update(), gr.update())

    framepack_target_resolution.change(
        fn=clear_explicit_res_on_target_change,
        inputs=[framepack_target_resolution],
        outputs=[framepack_width, framepack_height]
    )
    def toggle_f1_model_path(is_f1):
        f1_path = "hunyuan/FramePack_F1_I2V_HY_20250503.safetensors"
        standard_path = "hunyuan/FramePackI2V_HY_bf16.safetensors"
        target_path = f1_path if is_f1 else standard_path

        # Check if the target path exists
        if not os.path.exists(target_path):
             print(f"Warning: F1 model path '{target_path}' not found. Falling back to standard path.")
             # Optionally fall back or just update with the non-existent path
             # Let's fall back to standard if F1 is missing, but keep standard if standard is missing (error handled later)
             if is_f1 and os.path.exists(standard_path):
                 print(f"Falling back to standard path: {standard_path}")
                 return gr.update(value=standard_path)
             elif is_f1:
                 print(f"F1 path missing and standard path also missing. Cannot automatically switch.")
                 # Return the intended (missing) path, error will be caught later
                 return gr.update(value=target_path)
             else: # Standard path is missing
                  print(f"Warning: Standard path '{standard_path}' not found.")
                  return gr.update(value=target_path) # Return the missing standard path

        print(f"Switching DiT path to: {target_path}")
        return gr.update(value=target_path)

    framepack_is_f1.change(
        fn=toggle_f1_model_path,
        inputs=[framepack_is_f1],
        outputs=[framepack_transformer_path]
    )    

    framepack_generate_btn.click(
        fn=process_framepack_video,
        inputs=[
            # Standard args
            framepack_prompt, framepack_negative_prompt, framepack_input_image,
            framepack_input_video,
            # End Frame args
            framepack_input_end_frame, framepack_end_frame_influence, framepack_end_frame_weight,
            # Model Paths
            framepack_transformer_path, framepack_vae_path, framepack_text_encoder_path,
            framepack_text_encoder_2_path, framepack_image_encoder_path,
            # Core Params
            framepack_target_resolution, framepack_width, framepack_height, framepack_original_dims,
            framepack_total_second_length, framepack_fps, framepack_seed, framepack_steps,
            framepack_distilled_guidance_scale, framepack_guidance_scale, framepack_guidance_rescale,
            framepack_sample_solver, framepack_latent_window_size,
            # Performance/Memory
            framepack_fp8, framepack_fp8_scaled, framepack_fp8_llm,
            framepack_blocks_to_swap, framepack_bulk_decode, framepack_attn_mode,
            framepack_vae_chunk_size, framepack_vae_spatial_tile_sample_min_size,
            framepack_device,
            framepack_use_teacache,
            framepack_teacache_steps,
            framepack_teacache_thresh,
            # Batching & Saving
            framepack_batch_size, framepack_save_path,
            # LoRA Params
            framepack_lora_folder,
            framepack_enable_preview,
            framepack_preview_every_n_sections,
            framepack_is_f1,
            *framepack_secs, *framepack_sec_prompts, *framepack_sec_images,
            # LoRAs (actual components)
            *framepack_lora_weights, *framepack_lora_multipliers
        ],
        outputs=[
            framepack_output,           # Main gallery
            framepack_preview_output,   # Preview video player
            framepack_batch_progress,   # Status text
            framepack_progress_text     # Progress text
        ],
        queue=True
    )

    framepack_random_seed.click(
        fn=set_random_seed,
        inputs=None, 
        outputs=[framepack_seed] 
    )
    # Connect FramePack Stop button
    framepack_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    # Connect FramePack Gallery selection
    def handle_framepack_gallery_select(evt: gr.SelectData) -> int:
        return evt.index

    framepack_output.select(
        fn=handle_framepack_gallery_select,
        outputs=framepack_selected_index
    )

    def refresh_lora_dropdowns_simple(lora_folder: str) -> List[gr.update]:
        """Refreshes LoRA choices, always defaulting the selection to 'None'."""
        new_choices = get_lora_options(lora_folder)
        results = []
        print(f"Refreshing LoRA dropdowns. Found choices: {new_choices}") # Debug print
        for i in range(4): # Update all 4 slots
            results.extend([
                gr.update(choices=new_choices, value="None"), # Always reset value to None
                gr.update(value=1.0) # Reset multiplier
            ])
        return results
    
    # FramePack LoRA Refresh Button Handler
    framepack_lora_refresh_outputs = []
    for i in range(len(framepack_lora_weights)):
        framepack_lora_refresh_outputs.extend([framepack_lora_weights[i], framepack_lora_multipliers[i]])

    # <<< Modify this click handler >>>
    framepack_refresh_lora_btn.click(
        fn=refresh_lora_dropdowns_simple, # Use the new simplified function
        inputs=[framepack_lora_folder],   # Only needs the folder path as input
        outputs=framepack_lora_refresh_outputs # Still outputs updates to all 8 components
    )

    def toggle_fun_control(use_fun_control):
        """Toggle control video visibility and update task suffix"""
        # Only update visibility, don't try to set paths
        return gr.update(visible=use_fun_control)

    def update_task_for_funcontrol(use_fun_control, current_task):
        """Add or remove -FC suffix from task based on checkbox"""
        if use_fun_control:
            if not current_task.endswith("-FC"):
                if "i2v" in current_task:
                    return "i2v-14B-FC"
                elif "t2v" in current_task:
                    return "t2v-14B-FC"
            return current_task
        else:
            if current_task.endswith("-FC"):
                return current_task.replace("-FC", "")
            return current_task

    wanx_use_fun_control.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x), gr.update(visible=x), gr.update(visible=x)),
        inputs=[wanx_use_fun_control],
        outputs=[wanx_control_video, wanx_control_strength, wanx_control_start, wanx_control_end]
    )

    # Make task change update checkbox state
    def update_from_task(task):
        """Update Fun-Control checkbox and control video visibility based on task"""
        is_fun_control = "-FC" in task
        return gr.update(value=is_fun_control), gr.update(visible=is_fun_control)

    wanx_task.change(
        fn=update_from_task,
        inputs=[wanx_task],
        outputs=[wanx_use_fun_control, wanx_control_video]
    )
    wanx_enable_cfg_skip.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[wanx_enable_cfg_skip],
        outputs=[wanx_cfg_skip_options]
    )

    wanx_t2v_enable_cfg_skip.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[wanx_t2v_enable_cfg_skip],
        outputs=[wanx_t2v_cfg_skip_options]
    )

    wanx_v2v_enable_cfg_skip.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[wanx_v2v_enable_cfg_skip],
        outputs=[wanx_v2v_cfg_skip_options]
    )

    #WanX-v2v tab functions
    wanx_v2v_prompt.change(fn=count_prompt_tokens, inputs=wanx_v2v_prompt, outputs=wanx_v2v_token_counter)

    # Stop button handler
    wanx_v2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    # Video input handling
    wanx_v2v_input.change(
        fn=update_wanx_v2v_dimensions,
        inputs=[wanx_v2v_input],
        outputs=[wanx_v2v_original_dims, wanx_v2v_width, wanx_v2v_height]
    )

    # Flow shift recommendation button
    wanx_v2v_recommend_flow_btn.click(
        fn=recommend_wanx_flow_shift,
        inputs=[wanx_v2v_width, wanx_v2v_height],
        outputs=[wanx_v2v_flow_shift]
    )

    # Width/height calculation buttons
    wanx_v2v_calc_width_btn.click(
        fn=calculate_wanx_width,  # Reuse function from WanX tabs
        inputs=[wanx_v2v_height, wanx_v2v_original_dims],
        outputs=[wanx_v2v_width]
    )

    wanx_v2v_calc_height_btn.click(
        fn=calculate_wanx_height,  # Reuse function from WanX tabs
        inputs=[wanx_v2v_width, wanx_v2v_original_dims],
        outputs=[wanx_v2v_height]
    )

    # Scale slider handling for adjusting dimensions
    wanx_v2v_scale_slider.change(
        fn=update_wanx_from_scale,  # Reuse function from WanX tabs
        inputs=[wanx_v2v_scale_slider, wanx_v2v_original_dims],
        outputs=[wanx_v2v_width, wanx_v2v_height]
    )

    def change_to_wanx_v2v_tab():
        return gr.Tabs(selected=6) 

    def send_wanx_t2v_to_v2v_input(gallery, selected_index):
        """Send the selected WanX-t2v video to WanX-v2v input"""
        if gallery is None or not gallery:
            return None, None

        if selected_index is None and len(gallery) == 1:
            selected_index = 0

        if selected_index is None or selected_index >= len(gallery):
            return None, None

        # Get the video path
        item = gallery[selected_index]
        video_path = parse_video_path(item)

        return video_path, "Video sent from WanX-t2v tab"
    
    wanx_t2v_send_to_wanx_v2v_btn.click(
        fn=send_wanx_t2v_to_v2v_input,
        inputs=[wanx_t2v_output, wanx_t2v_selected_index],
        outputs=[wanx_v2v_input, wanx_v2v_batch_progress]
    ).then(
        fn=lambda prompt: prompt,
        inputs=[wanx_t2v_prompt],
        outputs=[wanx_v2v_prompt]
    ).then(
        fn=change_to_wanx_v2v_tab,
        inputs=None,
        outputs=[tabs]
    )

    # Send video from WanX-i2v to WanX-v2v
    wanx_i2v_send_to_wanx_v2v_btn.click(
        fn=send_wanx_t2v_to_v2v_input,  # Reuse the same function
        inputs=[wanx_output, wanx_i2v_selected_index],
        outputs=[wanx_v2v_input, wanx_v2v_batch_progress]
    ).then(
        fn=lambda prompt: prompt,
        inputs=[wanx_prompt],
        outputs=[wanx_v2v_prompt]
    ).then(
        fn=change_to_wanx_v2v_tab,
        inputs=None,
        outputs=[tabs]
    )

    # Update model paths when task changes
    def update_model_paths_for_task(task):
        if "1.3B" in task:
            return gr.update(value="wan/wan2.1_t2v_1.3B_fp16.safetensors")
        else:
            return gr.update(value="wan/wan2.1_t2v_14B_fp16.safetensors")

    wanx_v2v_task.change(
        fn=update_model_paths_for_task,
        inputs=[wanx_v2v_task],
        outputs=[wanx_v2v_dit_path]
    )

    # Generate button handler
    wanx_v2v_generate_btn.click(
        fn=wanx_v2v_batch_handler,
        inputs=[
            wanx_v2v_prompt,
            wanx_v2v_negative_prompt,
            wanx_v2v_input,
            wanx_v2v_width,
            wanx_v2v_height,
            wanx_v2v_video_length,
            wanx_v2v_fps,
            wanx_v2v_infer_steps,
            wanx_v2v_flow_shift,
            wanx_v2v_guidance_scale,
            wanx_v2v_strength,
            wanx_v2v_seed,
            wanx_v2v_batch_size,
            wanx_v2v_task,
            wanx_v2v_dit_folder,
            wanx_v2v_dit_path,
            wanx_v2v_vae_path,
            wanx_v2v_t5_path,
            wanx_v2v_save_path,
            wanx_v2v_output_type,
            wanx_v2v_sample_solver,
            wanx_v2v_exclude_single_blocks,
            wanx_v2v_attn_mode,
            wanx_v2v_block_swap,
            wanx_v2v_fp8,
            wanx_v2v_fp8_scaled,
            wanx_v2v_fp8_t5,
            wanx_v2v_lora_folder,
            wanx_v2v_slg_layers,
            wanx_v2v_slg_start,
            wanx_v2v_slg_end,
            wanx_v2v_enable_cfg_skip,
            wanx_v2v_cfg_skip_mode,
            wanx_v2v_cfg_apply_ratio,
            *wanx_v2v_lora_weights,
            *wanx_v2v_lora_multipliers
        ],
        outputs=[wanx_v2v_output, wanx_v2v_batch_progress, wanx_v2v_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[wanx_v2v_batch_size],
        outputs=wanx_v2v_selected_index
    )

    # Gallery selection handling
    wanx_v2v_output.select(
        fn=handle_wanx_v2v_gallery_select,
        outputs=wanx_v2v_selected_index
    )
    def change_to_tab_two():
        return gr.Tabs(selected=2)

    # Send to Hunyuan v2v tab
    wanx_v2v_send_to_v2v_btn.click(
        fn=send_wanx_v2v_to_hunyuan_v2v,
        inputs=[
            wanx_v2v_output,
            wanx_v2v_prompt,
            wanx_v2v_selected_index,
            wanx_v2v_width,
            wanx_v2v_height,
            wanx_v2v_video_length,
            wanx_v2v_fps,
            wanx_v2v_infer_steps,
            wanx_v2v_seed,
            wanx_v2v_flow_shift,
            wanx_v2v_guidance_scale,
            wanx_v2v_negative_prompt
        ],
        outputs=[
            v2v_input,
            v2v_prompt,
            v2v_width,
            v2v_height,
            v2v_video_length,
            v2v_fps,
            v2v_infer_steps,
            v2v_seed,
            v2v_flow_shift,
            v2v_cfg_scale,
            v2v_negative_prompt
        ]
    ).then(
        fn=change_to_tab_two,
        inputs=None,
        outputs=[tabs]
    )

    # Add refresh button handler for WanX-v2v tab
    wanx_v2v_refresh_outputs = [wanx_v2v_dit_path]  # This is one output
    for i in range(4):
        wanx_v2v_refresh_outputs.extend([wanx_v2v_lora_weights[i], wanx_v2v_lora_multipliers[i]])  # This adds 8 more outputs

    wanx_v2v_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,  # We need to use this function instead
        inputs=[wanx_v2v_dit_folder, wanx_v2v_lora_folder, wanx_v2v_dit_path] + wanx_v2v_lora_weights + wanx_v2v_lora_multipliers,
        outputs=wanx_v2v_refresh_outputs
    )

    # Add function to send videos from Video Info tab to WanX-v2v
    def send_to_wanx_v2v(metadata: dict, video_path: str) -> Tuple[str, Dict, str]:
        """Handle both parameters and video transfer from Video Info to WanX-v2v tab with debugging"""
        if not video_path:
            return "No video selected", {}, None

        # Print debug information
        print(f"VIDEO INFO TO WANX-V2V TRANSFER:")
        print(f"Original metadata: {metadata}")
        print(f"Video path: {video_path}")

        # Special handling for WanX-v2v prompt fields
        # Create a copy of metadata with explicit prompt fields
        enhanced_metadata = metadata.copy()
        if "prompt" in metadata:
            enhanced_metadata["wanx_v2v_prompt"] = metadata["prompt"]
        if "negative_prompt" in metadata:
            enhanced_metadata["wanx_v2v_negative_prompt"] = metadata["negative_prompt"]

        print(f"Enhanced metadata: {enhanced_metadata}")

        status_msg, params = send_parameters_to_tab(enhanced_metadata, "wanx_v2v")
        print(f"Mapped parameters: {params}")

        return f"Parameters ready for WanX-v2v (DEBUG INFO IN CONSOLE)", enhanced_metadata, video_path

    # Then, implement a proper handler to change to the WanX-v2v tab
    def change_to_wanx_v2v_tab():
        return gr.Tabs(selected=6)  # WanX-v2v is tab index 6

    # Next, connect the button to the functions with proper parameter mapping
    send_to_wanx_v2v_btn.click(
        fn=lambda m, v: handle_send_to_wanx_tab(m, 'wanx_v2v', v),
        inputs=[metadata_output, video_input],
        outputs=[status, params_state, wanx_v2v_input]
    ).then(
        lambda params: [
            params.get("prompt", ""),
            params.get("width", 832),
            params.get("height", 480),
            params.get("video_length", 81),
            params.get("fps", 16),
            params.get("infer_steps", 40),
            params.get("seed", -1),
            params.get("flow_shift", 5.0),
            params.get("guidance_scale", 5.0),
            params.get("attn_mode", "sdpa"),
            params.get("block_swap", 0),
            params.get("negative_prompt", ""),
            params.get("strength", 0.75),
            *[params.get("lora_weights", ["None"]*4)[i] if isinstance(params.get("lora_weights", []), list) and i < len(params.get("lora_weights", [])) else "None" for i in range(4)],
            *[params.get("lora_multipliers", [1.0]*4)[i] if isinstance(params.get("lora_multipliers", []), list) and i < len(params.get("lora_multipliers", [])) else 1.0 for i in range(4)]
        ] if params else [gr.update()]*21,
        inputs=params_state,
        outputs=[
            wanx_v2v_prompt,
            wanx_v2v_width,
            wanx_v2v_height,
            wanx_v2v_video_length,
            wanx_v2v_fps,
            wanx_v2v_infer_steps,
            wanx_v2v_seed,
            wanx_v2v_flow_shift,
            wanx_v2v_guidance_scale,
            wanx_v2v_attn_mode,
            wanx_v2v_block_swap,
            wanx_v2v_negative_prompt,
            wanx_v2v_strength,
            *wanx_v2v_lora_weights,
            *wanx_v2v_lora_multipliers
        ]
    ).then(
        fn=change_to_wanx_v2v_tab, inputs=None, outputs=[tabs]
    )

    #Video Extension
    wanx_send_last_frame_btn.click(
        fn=send_last_frame_handler,
        inputs=[wanx_output, wanx_i2v_selected_index],
        outputs=[wanx_input, wanx_base_video]
    )

    wanx_extend_btn.click(
        fn=prepare_for_batch_extension,
        inputs=[wanx_input, wanx_base_video, wanx_batch_size],
        outputs=[wanx_input, wanx_base_video, wanx_batch_size, wanx_batch_progress, wanx_progress_text]
    ).then(
        fn=lambda batch_size, base_video:
            "Starting batch extension..." if base_video and batch_size > 0 else
            "Error: Missing base video or invalid batch size",
        inputs=[wanx_batch_size, wanx_base_video],
        outputs=[wanx_batch_progress]
    ).then(
        # Process batch extension one at a time
        fn=process_batch_extension,
        inputs=[
            wanx_prompt,
            wanx_negative_prompt,
            wanx_input,               # Input image (last frame)
            wanx_base_video,          # Base video to extend
            wanx_width,
            wanx_height,
            wanx_video_length,
            wanx_fps,
            wanx_infer_steps,
            wanx_flow_shift,
            wanx_guidance_scale,
            wanx_seed,
            wanx_batch_size,
            wanx_task,
            wanx_dit_folder,          # <<< Pass the folder path
            wanx_dit_path,            # <<< Pass the model filename
            wanx_vae_path,
            wanx_t5_path,
            wanx_clip_path,
            wanx_save_path,
            wanx_output_type,
            wanx_sample_solver,
            wanx_exclude_single_blocks,
            wanx_attn_mode,
            wanx_block_swap,
            wanx_fp8,
            wanx_fp8_scaled,
            wanx_fp8_t5,
            wanx_lora_folder,
            wanx_slg_layers,
            wanx_slg_start,
            wanx_slg_end,
            # Pass LoRA weights and multipliers individually
            wanx_lora_weights[0],
            wanx_lora_weights[1],
            wanx_lora_weights[2],
            wanx_lora_weights[3],
            wanx_lora_multipliers[0],
            wanx_lora_multipliers[1],
            wanx_lora_multipliers[2],
            wanx_lora_multipliers[3]
        ],
        outputs=[wanx_output, wanx_batch_progress, wanx_progress_text]
    )

    # Extract and send sharpest frame to input
    wanx_send_sharpest_frame_btn.click(
        fn=send_sharpest_frame_handler,
        inputs=[wanx_output, wanx_i2v_selected_index, wanx_frames_to_check],
        outputs=[wanx_input, wanx_base_video, wanx_sharpest_frame_number, wanx_sharpest_frame_status]
    )

    # Trim video to sharpest frame and prepare for extension
    wanx_trim_and_extend_btn.click(
        fn=trim_and_prepare_for_extension,
        inputs=[wanx_base_video, wanx_sharpest_frame_number, wanx_save_path],
        outputs=[wanx_trimmed_video_path, wanx_sharpest_frame_status]
    ).then(
        fn=lambda path, status: (path, status if "Failed" in status else "Video trimmed successfully and ready for extension"),
        inputs=[wanx_trimmed_video_path, wanx_sharpest_frame_status],
        outputs=[wanx_base_video, wanx_sharpest_frame_status]
    )

    wanx_extend_with_trimmed_btn.click(
        # Prepare step: Sets the base video to the trimmed video path
        fn=prepare_for_batch_extension,
        inputs=[wanx_input, wanx_trimmed_video_path, wanx_batch_size], # Use trimmed video path here
        outputs=[wanx_input, wanx_base_video, wanx_batch_size, wanx_batch_progress, wanx_progress_text] # Update base_video state
    ).then(
        # Actual extension processing step
        fn=process_batch_extension,
        inputs=[
            wanx_prompt,
            wanx_negative_prompt,
            wanx_input,               # Input image (sharpest frame)
            wanx_trimmed_video_path,  # Base video to extend (the trimmed one)
            wanx_width,
            wanx_height,
            wanx_video_length,
            wanx_fps,
            wanx_infer_steps,
            wanx_flow_shift,
            wanx_guidance_scale,
            wanx_seed,
            wanx_batch_size,
            wanx_task,
            wanx_dit_folder,          # <<< Pass the folder path
            wanx_dit_path,            # <<< Pass the model filename
            wanx_vae_path,
            wanx_t5_path,
            wanx_clip_path,
            wanx_save_path,
            wanx_output_type,
            wanx_sample_solver,
            wanx_exclude_single_blocks,
            wanx_attn_mode,
            wanx_block_swap,
            wanx_fp8,
            wanx_fp8_scaled,
            wanx_fp8_t5,
            wanx_lora_folder,
            wanx_slg_layers,
            wanx_slg_start,
            wanx_slg_end,
            # Pass LoRA weights and multipliers individually
            wanx_lora_weights[0],
            wanx_lora_weights[1],
            wanx_lora_weights[2],
            wanx_lora_weights[3],
            wanx_lora_multipliers[0],
            wanx_lora_multipliers[1],
            wanx_lora_multipliers[2],
            wanx_lora_multipliers[3]
        ],
        outputs=[wanx_output, wanx_batch_progress, wanx_progress_text]
    )

    #Video Info
    def handle_send_to_wanx_tab(metadata, target_tab, video_path=None):
        """Common handler for sending video parameters to WanX tabs"""
        if not metadata:
            return "No parameters to send", {}, None  # Return three values
    
        # Tab names for clearer messages
        tab_names = {
            'wanx_i2v': 'WanX-i2v',
            'wanx_t2v': 'WanX-t2v',
            'wanx_v2v': 'WanX-v2v'
        }
    
        # Just pass through all parameters - we'll use them in the .then() function
        return f"Parameters ready for {tab_names.get(target_tab, target_tab)}", metadata, video_path

    def change_to_wanx_i2v_tab():
        return gr.Tabs(selected=4)  # WanX-i2v tab index

    def change_to_wanx_t2v_tab():
        return gr.Tabs(selected=5)  # WanX-t2v tab index


    send_to_wanx_i2v_btn.click(
        fn=lambda m: ("Parameters ready for WanX-i2v", m),
        inputs=[metadata_output],
        outputs=[status, params_state]
    ).then(
        # Reusing the same pattern as other tab transfers with LoRA handling
        lambda params: [
            params.get("prompt", ""),
            params.get("width", 832),
            params.get("height", 480),
            params.get("video_length", 81),
            params.get("fps", 16),
            params.get("infer_steps", 40),
            params.get("seed", -1),
            params.get("flow_shift", 3.0),
            params.get("guidance_scale", 5.0),
            params.get("attn_mode", "sdpa"),
            params.get("block_swap", 0),
            params.get("task", "i2v-14B"),
            params.get("negative_prompt", ""),
            *[params.get("lora_weights", ["None"]*4)[i] if isinstance(params.get("lora_weights", []), list) and i < len(params.get("lora_weights", [])) else "None" for i in range(4)],
            *[params.get("lora_multipliers", [1.0]*4)[i] if isinstance(params.get("lora_multipliers", []), list) and i < len(params.get("lora_multipliers", [])) else 1.0 for i in range(4)]
        ] if params else [gr.update()]*20,
        inputs=params_state,
        outputs=[
            wanx_prompt, wanx_width, wanx_height, wanx_video_length, 
            wanx_fps, wanx_infer_steps, wanx_seed, wanx_flow_shift, 
            wanx_guidance_scale, wanx_attn_mode, wanx_block_swap,
            wanx_task, wanx_negative_prompt, 
            *wanx_lora_weights,
            *wanx_lora_multipliers
        ]
    ).then(
        fn=change_to_wanx_i2v_tab, 
        inputs=None, 
        outputs=[tabs]
    )

    # 3. Update the WanX-t2v button handler
    send_to_wanx_t2v_btn.click(
        fn=lambda m: handle_send_to_wanx_tab(m, 'wanx_t2v'),
        inputs=[metadata_output],
        outputs=[status, params_state]
    ).then(
        lambda params: [
            params.get("prompt", ""),
            params.get("width", 832),
            params.get("height", 480),
            params.get("video_length", 81),
            params.get("fps", 16),
            params.get("infer_steps", 50),
            params.get("seed", -1),
            params.get("flow_shift", 5.0),
            params.get("guidance_scale", 5.0),
            params.get("attn_mode", "sdpa"),
            params.get("block_swap", 0),
            params.get("negative_prompt", ""),
            *[params.get("lora_weights", ["None"]*4)[i] if isinstance(params.get("lora_weights", []), list) and i < len(params.get("lora_weights", [])) else "None" for i in range(4)],
            *[params.get("lora_multipliers", [1.0]*4)[i] if isinstance(params.get("lora_multipliers", []), list) and i < len(params.get("lora_multipliers", [])) else 1.0 for i in range(4)]
        ] if params else [gr.update()]*20,
        inputs=params_state,
        outputs=[
            wanx_t2v_prompt,
            wanx_t2v_width,
            wanx_t2v_height,
            wanx_t2v_video_length,
            wanx_t2v_fps,
            wanx_t2v_infer_steps,
            wanx_t2v_seed,
            wanx_t2v_flow_shift,
            wanx_t2v_guidance_scale,
            wanx_t2v_attn_mode,
            wanx_t2v_block_swap,
            wanx_t2v_negative_prompt,
            *wanx_t2v_lora_weights,
            *wanx_t2v_lora_multipliers
        ]
    ).then(
        fn=change_to_wanx_t2v_tab, inputs=None, outputs=[tabs]
    )

    #text to video
    def change_to_tab_one():
        return gr.Tabs(selected=1) #This will navigate
    #video to video

    def change_to_skyreels_tab():
        return gr.Tabs(selected=3) 
    
    #SKYREELS TAB!!!
    # Add state management for dimensions
    def sync_skyreels_dimensions(width, height):
        return gr.update(value=width), gr.update(value=height)

    # Add this function to update the LoRA dropdowns in the SKYREELS tab
    def update_skyreels_lora_dropdowns(lora_folder: str, *current_values) -> List[gr.update]:
        new_choices = get_lora_options(lora_folder)
        weights = current_values[:4]
        multipliers = current_values[4:8]

        results = []
        for i in range(4):
            weight = weights[i] if i < len(weights) else "None"
            multiplier = multipliers[i] if i < len(multipliers) else 1.0
            if weight not in new_choices:
                weight = "None"
            results.extend([
                gr.update(choices=new_choices, value=weight),
                gr.update(value=multiplier) 
            ])

        return results

    # Add this function to update the models dropdown in the SKYREELS tab
    def update_skyreels_model_dropdown(dit_folder: str) -> Dict:
        models = get_dit_models(dit_folder)
        return gr.update(choices=models, value=models[0] if models else None)

    # Add event handler for model dropdown refresh
    skyreels_dit_folder.change(
        fn=update_skyreels_model_dropdown,
        inputs=[skyreels_dit_folder],
        outputs=[skyreels_model]
    )

    # Add handlers for the refresh button
    skyreels_refresh_btn.click(
        fn=update_skyreels_lora_dropdowns,
        inputs=[skyreels_lora_folder] + skyreels_lora_weights + skyreels_lora_multipliers,
        outputs=[drop for _ in range(4) for drop in [skyreels_lora_weights[_], skyreels_lora_multipliers[_]]]
    )      
    # Skyreels dimension handling
    def calculate_skyreels_width(height, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_width = math.floor((height * aspect_ratio) / 16) * 16
        return gr.update(value=new_width)

    def calculate_skyreels_height(width, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_height = math.floor((width / aspect_ratio) / 16) * 16
        return gr.update(value=new_height)

    def update_skyreels_from_scale(scale, original_dims):
        if not original_dims:
            return gr.update(), gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        new_w = math.floor((orig_w * scale / 100) / 16) * 16
        new_h = math.floor((orig_h * scale / 100) / 16) * 16
        return gr.update(value=new_w), gr.update(value=new_h)

    def update_skyreels_dimensions(image):
        if image is None:
            return "", gr.update(value=544), gr.update(value=544)
        img = Image.open(image)
        w, h = img.size
        w = (w // 16) * 16
        h = (h // 16) * 16
        return f"{w}x{h}", w, h

    def handle_skyreels_gallery_select(evt: gr.SelectData) -> int:
        return evt.index

    def send_skyreels_to_v2v(
        gallery: list,
        prompt: str,
        selected_index: int,
        width: int,
        height: int,
        video_length: int,
        fps: int,
        infer_steps: int,
        seed: int,
        flow_shift: float,
        cfg_scale: float,
        lora1: str,
        lora2: str,
        lora3: str,
        lora4: str,
        lora1_multiplier: float,
        lora2_multiplier: float,
        lora3_multiplier: float,
        lora4_multiplier: float,
        negative_prompt: str = ""  # Add this parameter
    ) -> Tuple:
        if not gallery or selected_index is None or selected_index >= len(gallery):
            return (None, "", width, height, video_length, fps, infer_steps, seed, 
                    flow_shift, cfg_scale, lora1, lora2, lora3, lora4,
                    lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier,
                    negative_prompt)  # Add negative_prompt to return

        selected_item = gallery[selected_index]

        if isinstance(selected_item, dict):
            video_path = selected_item.get("name", selected_item.get("data", None))
        elif isinstance(selected_item, (tuple, list)):
            video_path = selected_item[0]
        else:
            video_path = selected_item

        if isinstance(video_path, tuple):
            video_path = video_path[0]

        return (str(video_path), prompt, width, height, video_length, fps, infer_steps, seed, 
                flow_shift, cfg_scale, lora1, lora2, lora3, lora4,
                lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier,
                negative_prompt)  # Add negative_prompt to return

    # Add event handlers for the SKYREELS tab
    skyreels_prompt.change(fn=count_prompt_tokens, inputs=skyreels_prompt, outputs=skyreels_token_counter)
    skyreels_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    # Image input handling
    skyreels_input.change(
        fn=update_skyreels_dimensions,
        inputs=[skyreels_input],
        outputs=[skyreels_original_dims, skyreels_width, skyreels_height]
    )

    skyreels_scale_slider.change(
        fn=update_skyreels_from_scale,
        inputs=[skyreels_scale_slider, skyreels_original_dims],
        outputs=[skyreels_width, skyreels_height]
    )

    skyreels_calc_width_btn.click(
        fn=calculate_skyreels_width,
        inputs=[skyreels_height, skyreels_original_dims],
        outputs=[skyreels_width]
    )

    skyreels_calc_height_btn.click(
        fn=calculate_skyreels_height,
        inputs=[skyreels_width, skyreels_original_dims],
        outputs=[skyreels_height]
    )

    # Handle checkbox visibility toggling
    skyreels_use_random_folder.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x), gr.update(visible=not x)),
        inputs=[skyreels_use_random_folder],
        outputs=[skyreels_input_folder, skyreels_folder_status, skyreels_input]
    )

    # Validate folder button click handler
    skyreels_validate_folder_btn.click(
        fn=lambda folder: get_random_image_from_folder(folder)[1],
        inputs=[skyreels_input_folder],
        outputs=[skyreels_folder_status]
    )

    skyreels_use_random_folder.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[skyreels_use_random_folder],
        outputs=[skyreels_validate_folder_btn]
    )

    # Modify the skyreels_generate_btn.click event handler to use process_random_image_batch when folder mode is on
    skyreels_generate_btn.click(
        fn=batch_handler,
        inputs=[
            skyreels_use_random_folder,
            # Rest of the arguments
            skyreels_prompt,
            skyreels_negative_prompt,
            skyreels_width,
            skyreels_height,
            skyreels_video_length,
            skyreels_fps,
            skyreels_infer_steps,
            skyreels_seed,
            skyreels_flow_shift,
            skyreels_guidance_scale,
            skyreels_embedded_cfg_scale,
            skyreels_batch_size,
            skyreels_input_folder,
            skyreels_dit_folder,
            skyreels_model,
            skyreels_vae,
            skyreels_te1,
            skyreels_te2,
            skyreels_save_path,
            skyreels_output_type,
            skyreels_attn_mode,
            skyreels_block_swap,
            skyreels_exclude_single_blocks,
            skyreels_use_split_attn,
            skyreels_use_fp8,
            skyreels_split_uncond,
            skyreels_lora_folder,
            *skyreels_lora_weights,
            *skyreels_lora_multipliers,
            skyreels_input  # Add the input image path
        ],
        outputs=[skyreels_output, skyreels_batch_progress, skyreels_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[skyreels_batch_size],
        outputs=skyreels_selected_index
    )

    # Gallery selection handling
    skyreels_output.select(
        fn=handle_skyreels_gallery_select,
        outputs=skyreels_selected_index
    )

    # Send to Video2Video handler
    skyreels_send_to_v2v_btn.click(
        fn=send_skyreels_to_v2v,
        inputs=[
            skyreels_output, skyreels_prompt, skyreels_selected_index,
            skyreels_width, skyreels_height, skyreels_video_length,
            skyreels_fps, skyreels_infer_steps, skyreels_seed,
            skyreels_flow_shift, skyreels_guidance_scale
        ] + skyreels_lora_weights + skyreels_lora_multipliers + [skyreels_negative_prompt],  # This is ok because skyreels_negative_prompt is a Gradio component
        outputs=[
            v2v_input, v2v_prompt, v2v_width, v2v_height,
            v2v_video_length, v2v_fps, v2v_infer_steps,
            v2v_seed, v2v_flow_shift, v2v_cfg_scale
        ] + v2v_lora_weights + v2v_lora_multipliers + [v2v_negative_prompt]
    ).then(
        fn=change_to_tab_two,
        inputs=None,
        outputs=[tabs]
    )

    # Refresh button handler
    skyreels_refresh_outputs = [skyreels_model]
    for i in range(4):
        skyreels_refresh_outputs.extend([skyreels_lora_weights[i], skyreels_lora_multipliers[i]])

    skyreels_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,
        inputs=[skyreels_dit_folder, skyreels_lora_folder, skyreels_model] + skyreels_lora_weights + skyreels_lora_multipliers,
        outputs=skyreels_refresh_outputs
    )
      
    def calculate_v2v_width(height, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_width = math.floor((height * aspect_ratio) / 16) * 16  # Ensure divisible by 16
        return gr.update(value=new_width)

    def calculate_v2v_height(width, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_height = math.floor((width / aspect_ratio) / 16) * 16  # Ensure divisible by 16
        return gr.update(value=new_height)

    def update_v2v_from_scale(scale, original_dims):
        if not original_dims:
            return gr.update(), gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        new_w = math.floor((orig_w * scale / 100) / 16) * 16  # Ensure divisible by 16
        new_h = math.floor((orig_h * scale / 100) / 16) * 16  # Ensure divisible by 16
        return gr.update(value=new_w), gr.update(value=new_h)

    def update_v2v_dimensions(video):
        if video is None:
            return "", gr.update(value=544), gr.update(value=544)
        cap = cv2.VideoCapture(video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        # Make dimensions divisible by 16
        w = (w // 16) * 16
        h = (h // 16) * 16
        return f"{w}x{h}", w, h
    
    # Event Handlers for Video to Video Tab
    v2v_input.change(
        fn=update_v2v_dimensions,
        inputs=[v2v_input],
        outputs=[v2v_original_dims, v2v_width, v2v_height]
    )

    v2v_scale_slider.change(
        fn=update_v2v_from_scale,
        inputs=[v2v_scale_slider, v2v_original_dims],
        outputs=[v2v_width, v2v_height]
    )

    v2v_calc_width_btn.click(
        fn=calculate_v2v_width,
        inputs=[v2v_height, v2v_original_dims],
        outputs=[v2v_width]
    )

    v2v_calc_height_btn.click(
        fn=calculate_v2v_height,
        inputs=[v2v_width, v2v_original_dims],
        outputs=[v2v_height]
    )

    ##Image 2 video dimension logic
    def calculate_width(height, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_width = math.floor((height * aspect_ratio) / 16) * 16  # Changed from 8 to 16
        return gr.update(value=new_width)

    def calculate_height(width, original_dims):
        if not original_dims:
            return gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        aspect_ratio = orig_w / orig_h
        new_height = math.floor((width / aspect_ratio) / 16) * 16  # Changed from 8 to 16
        return gr.update(value=new_height)

    def update_from_scale(scale, original_dims):
        if not original_dims:
            return gr.update(), gr.update()
        orig_w, orig_h = map(int, original_dims.split('x'))
        new_w = math.floor((orig_w * scale / 100) / 16) * 16  # Changed from 8 to 16
        new_h = math.floor((orig_h * scale / 100) / 16) * 16  # Changed from 8 to 16
        return gr.update(value=new_w), gr.update(value=new_h)

    def update_dimensions(image):
        if image is None:
            return "", gr.update(value=544), gr.update(value=544)
        img = Image.open(image)
        w, h = img.size
        # Make dimensions divisible by 16
        w = (w // 16) * 16  # Changed from 8 to 16
        h = (h // 16) * 16  # Changed from 8 to 16
        return f"{w}x{h}", w, h
    i2v_input.change(
        fn=update_dimensions,
        inputs=[i2v_input],
        outputs=[original_dims, i2v_width, i2v_height] # Update correct components
    )

    scale_slider.change(
        fn=update_from_scale,
        inputs=[scale_slider, original_dims],
        outputs=[i2v_width, i2v_height] # Update correct components
    )

    calc_width_btn.click(
        fn=calculate_width,
        inputs=[i2v_height, original_dims], # Update correct components
        outputs=[i2v_width]
    )

    calc_height_btn.click(
        fn=calculate_height,
        inputs=[i2v_width, original_dims], # Update correct components
        outputs=[i2v_height]    
    )

    # Function to get available DiT models
    def get_dit_models(dit_folder: str) -> List[str]:
        if not os.path.exists(dit_folder):
            return ["mp_rank_00_model_states.pt"]
        models = [f for f in os.listdir(dit_folder) if f.endswith('.pt') or f.endswith('.safetensors')]
        models.sort(key=str.lower)
        return models if models else ["mp_rank_00_model_states.pt"]

    # Function to perform model merging
    def merge_models(
        dit_folder: str,
        dit_model: str,
        output_model: str,
        exclude_single_blocks: bool,
        merge_lora_folder: str,
        *lora_params  # Will contain both weights and multipliers
    ) -> str:
        try:
            # Separate weights and multipliers
            num_loras = len(lora_params) // 2
            weights = list(lora_params[:num_loras])
            multipliers = list(lora_params[num_loras:])

            # Filter out "None" selections
            valid_loras = []
            for weight, mult in zip(weights, multipliers):
                if weight and weight != "None":
                    valid_loras.append((os.path.join(merge_lora_folder, weight), mult))

            if not valid_loras:
                return "No LoRA models selected for merging"

            # Create output path in the dit folder
            os.makedirs(dit_folder, exist_ok=True)
            output_path = os.path.join(dit_folder, output_model)
            
            # Prepare command
            cmd = [
                sys.executable,
                "merge_lora.py",
                "--dit", os.path.join(dit_folder, dit_model),
                "--save_merged_model", output_path
            ]

            # Add LoRA weights and multipliers
            weights = [weight for weight, _ in valid_loras]
            multipliers = [str(mult) for _, mult in valid_loras]
            cmd.extend(["--lora_weight"] + weights)
            cmd.extend(["--lora_multiplier"] + multipliers)

            if exclude_single_blocks:
                cmd.append("--exclude_single_blocks")

            # Execute merge operation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            if os.path.exists(output_path):
                return f"Successfully merged model and saved to {output_path}"
            else:
                return "Error: Output file not created"

        except subprocess.CalledProcessError as e:
            return f"Error during merging: {e.stderr}"
        except Exception as e:
            return f"Error: {str(e)}"

    # Update DiT model dropdown
    def update_dit_dropdown(dit_folder: str) -> Dict:
        models = get_dit_models(dit_folder)
        return gr.update(choices=models, value=models[0] if models else None)

    # Connect events
    merge_btn.click(
        fn=merge_models,
        inputs=[
            dit_folder,
            dit_model,
            output_model,
            exclude_single_blocks,
            merge_lora_folder,
            *merge_lora_weights,
            *merge_lora_multipliers
        ],
        outputs=merge_status
    )

    # Refresh buttons for both DiT and LoRA dropdowns
    merge_refresh_btn.click(
        fn=lambda f: update_dit_dropdown(f),
        inputs=[dit_folder],
        outputs=[dit_model]
    )

    # LoRA refresh handling
    merge_refresh_outputs = []
    for i in range(4):
        merge_refresh_outputs.extend([merge_lora_weights[i], merge_lora_multipliers[i]])

    merge_refresh_btn.click(
        fn=update_lora_dropdowns,
        inputs=[merge_lora_folder] + merge_lora_weights + merge_lora_multipliers,
        outputs=merge_refresh_outputs
    )
    # Event handlers
    prompt.change(fn=count_prompt_tokens, inputs=prompt, outputs=token_counter)
    v2v_prompt.change(fn=count_prompt_tokens, inputs=v2v_prompt, outputs=v2v_token_counter)
    stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    v2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    #Image_to_Video
    def image_to_video(image_path, output_path, width, height, frames=240):  # Add width, height parameters
        img = Image.open(image_path)

        # Resize to the specified dimensions
        img_resized = img.resize((width, height), Image.LANCZOS)
        temp_image_path = os.path.join(os.path.dirname(output_path), "temp_resized_image.png")
        img_resized.save(temp_image_path)

        # Rest of function remains the same
        frame_rate = 24
        duration = frames / frame_rate
        command = [
            "ffmpeg", "-loop", "1", "-i", temp_image_path, "-c:v", "libx264",
            "-t", str(duration), "-pix_fmt", "yuv420p",
            "-vf", f"fps={frame_rate}", output_path
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Video saved to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while creating the video: {e}")
            return False
        finally:
            # Clean up the temporary image file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            img.close()  # Make sure to close the image file explicitly

    def generate_from_image(
        image_path, 
        prompt, width, height, video_length, fps, infer_steps,
        seed, model, vae, te1, te2, save_path, flow_shift, cfg_scale, 
        output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
        lora_folder, strength, batch_size, *lora_params
    ):
        """Generate video from input image with progressive updates"""
        global stop_event
        stop_event.clear()
    
        # Create temporary video path
        temp_video_path = os.path.join(save_path, f"temp_{os.path.basename(image_path)}.mp4")
    
        try:
            # Convert image to video
            if not image_to_video(image_path, temp_video_path, width, height, frames=video_length):
                yield [], "Failed to create temporary video", "Error in video creation"
                return
    
            # Ensure video is fully written before proceeding
            time.sleep(1)
            if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                yield [], "Failed to create temporary video", "Temporary video file is empty or missing"
                return
    
            # Get video dimensions
            try:
                probe = ffmpeg.probe(temp_video_path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if video_stream is None:
                    raise ValueError("No video stream found")
                width = int(video_stream['width'])
                height = int(video_stream['height'])
            except Exception as e:
                yield [], f"Error reading video dimensions: {str(e)}", "Video processing error"
                return
    
            # Generate the video using the temporary file
            try:
                generator = process_single_video(
                    prompt, width, height, batch_size, video_length, fps, infer_steps,
                    seed, model, vae, te1, te2, save_path, flow_shift, cfg_scale,
                    output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
                    lora_folder, *lora_params, video_path=temp_video_path, strength=strength
                )
    
                # Forward all generator updates
                for videos, batch_text, progress_text in generator:
                    yield videos, batch_text, progress_text
    
            except Exception as e:
                yield [], f"Error in video generation: {str(e)}", "Generation error"
                return
    
        except Exception as e:
            yield [], f"Unexpected error: {str(e)}", "Error occurred"
            return
    
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            except Exception:
                pass  # Ignore cleanup errors


    # Add event handlers
    i2v_prompt.change(fn=count_prompt_tokens, inputs=i2v_prompt, outputs=i2v_token_counter)
    i2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    def handle_i2v_gallery_select(evt: gr.SelectData) -> int:
        """Track selected index when I2V gallery item is clicked"""
        return evt.index

    def send_i2v_to_v2v(
        gallery: list, 
        prompt: str, 
        selected_index: int,
        width: int,
        height: int,
        video_length: int,
        fps: int,
        infer_steps: int,
        seed: int,
        flow_shift: float,
        cfg_scale: float,
        lora1: str,
        lora2: str,
        lora3: str,
        lora4: str,
        lora1_multiplier: float,
        lora2_multiplier: float,
        lora3_multiplier: float,
        lora4_multiplier: float
    ) -> Tuple[Optional[str], str, int, int, int, int, int, int, float, float, str, str, str, str, float, float, float, float]:
        """Send the selected video and parameters from Image2Video tab to Video2Video tab"""
        if not gallery or selected_index is None or selected_index >= len(gallery):
            return None, "", width, height, video_length, fps, infer_steps, seed, flow_shift, cfg_scale, \
                   lora1, lora2, lora3, lora4, lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier

        selected_item = gallery[selected_index]

        # Handle different gallery item formats
        if isinstance(selected_item, dict):
            video_path = selected_item.get("name", selected_item.get("data", None))
        elif isinstance(selected_item, (tuple, list)):
            video_path = selected_item[0]
        else:
            video_path = selected_item

        # Final cleanup for Gradio Video component
        if isinstance(video_path, tuple):
            video_path = video_path[0]

        # Use the original width and height without doubling
        return (str(video_path), prompt, width, height, video_length, fps, infer_steps, seed, 
                flow_shift, cfg_scale, lora1, lora2, lora3, lora4, 
                lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier)

    # Generate button handler for h-basic-i2v
    i2v_generate_btn.click(
        fn=process_i2v_batch, # <<< Use the new batch function
        inputs=[
            i2v_prompt,
            i2v_input, # Image path
            i2v_width,
            i2v_height,
            i2v_batch_size,
            i2v_video_length,
            i2v_fps,
            i2v_infer_steps,
            i2v_seed,
            i2v_dit_folder,
            i2v_model,
            i2v_vae,
            i2v_te1,
            i2v_te2,
            i2v_clip_vision_path,
            i2v_save_path,
            i2v_flow_shift,
            i2v_cfg_scale, # embedded_cfg_scale
            i2v_guidance_scale, # main CFG scale
            i2v_output_type,
            i2v_attn_mode,
            i2v_block_swap,
            i2v_exclude_single_blocks,
            i2v_use_split_attn,
            i2v_lora_folder,
            i2v_vae_chunk_size,
            i2v_vae_spatial_tile_min,
            # --- Add negative prompt component if you have one ---
            # i2v_negative_prompt, # Uncomment if you added this textbox
            # --- If no negative prompt textbox, pass None or "": ---
            gr.Textbox(value="", visible=False), # Placeholder if no UI element
            # --- End negative prompt handling ---
            i2v_use_fp8,
            i2v_fp8_llm,
            *i2v_lora_weights, # Pass LoRA weights components
            *i2v_lora_multipliers # Pass LoRA multipliers components
        ],
        outputs=[i2v_output, i2v_batch_progress, i2v_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[i2v_batch_size],
        outputs=i2v_selected_index
    )
    # Send to Video2Video
    i2v_output.select(
        fn=handle_i2v_gallery_select,
        outputs=i2v_selected_index
    )

    i2v_send_to_v2v_btn.click(
        fn=send_i2v_to_v2v, # Function definition needs careful review/update if args changed
        inputs=[
            i2v_output, i2v_prompt, i2v_selected_index,
            i2v_width, i2v_height, # <<< Use i2v width/height
            i2v_video_length, i2v_fps, i2v_infer_steps,
            i2v_seed, i2v_flow_shift, i2v_cfg_scale # <<< Use i2v cfg_scale (embedded)
        ] + i2v_lora_weights + i2v_lora_multipliers, # <<< Use i2v LoRAs
        outputs=[
            v2v_input, v2v_prompt,
            v2v_width, v2v_height, # Target V2V components
            v2v_video_length, v2v_fps, v2v_infer_steps,
            v2v_seed, v2v_flow_shift, v2v_cfg_scale # Target V2V components
        ] + v2v_lora_weights + v2v_lora_multipliers # Target V2V LoRAs
    ).then(
        fn=change_to_tab_two, inputs=None, outputs=[tabs]
    )
    #Video Info
    def clean_video_path(video_path) -> str:
        """Extract clean video path from Gradio's various return formats"""
        print(f"Input video_path: {video_path}, type: {type(video_path)}")
        if isinstance(video_path, dict):
            path = video_path.get("name", "")
        elif isinstance(video_path, (tuple, list)):
            path = video_path[0]
        elif isinstance(video_path, str):
            path = video_path
        else:
            path = ""
        print(f"Cleaned path: {path}")
        return path
    def handle_video_upload(video_path: str) -> Dict:
        """Handle video upload and metadata extraction"""
        if not video_path:
            return {}, "No video uploaded"

        metadata = extract_video_metadata(video_path)
        if not metadata:
            return {}, "No metadata found in video"

        return metadata, "Metadata extracted successfully"
    
    def get_video_info(video_path: str) -> dict:
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
            
            width = int(video_info['width'])
            height = int(video_info['height'])
            fps = eval(video_info['r_frame_rate'])  # This converts '30/1' to 30.0
            
            # Calculate total frames
            duration = float(probe['format']['duration'])
            total_frames = int(duration * fps)
            
            # Ensure video length does not exceed 201 frames
            if total_frames > 201:
                total_frames = 201
                duration = total_frames / fps  # Adjust duration accordingly
    
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration  # Might be useful in some contexts
            }
        except Exception as e:
            print(f"Error extracting video info: {e}")
            return {}
        
    def extract_video_details(video_path: str) -> Tuple[dict, str]:
        metadata = extract_video_metadata(video_path)
        video_details = get_video_info(video_path)

        # Combine metadata with video details
        for key, value in video_details.items():
            if key not in metadata:
                metadata[key] = value

        # Ensure video length does not exceed 201 frames
        if 'video_length' in metadata:
            metadata['video_length'] = min(metadata['video_length'], 201)
        else:
            metadata['video_length'] = min(video_details.get('total_frames', 0), 201)

        # Return both the updated metadata and a status message
        return metadata, "Video details extracted successfully"

    def send_parameters_to_tab(metadata: Dict, target_tab: str) -> Tuple[str, Dict]:
        """Create parameter mapping for target tab"""
        if not metadata:
            return "No parameters to send", {}

        tab_name = "Text2Video" if target_tab == "t2v" else "Video2Video"
        try:
            mapping = create_parameter_transfer_map(metadata, target_tab)
            return f"Parameters ready for {tab_name}", mapping
        except Exception as e:
            return f"Error: {str(e)}", {}
        
    video_input.upload(
        fn=extract_video_details,
        inputs=video_input,
        outputs=[metadata_output, status]
    )

    send_to_t2v_btn.click(
        fn=lambda m: send_parameters_to_tab(m, "t2v"),
        inputs=metadata_output,
        outputs=[status, params_state]
    ).then(
        fn=change_to_tab_one, inputs=None, outputs=[tabs]
    ).then(
        lambda params: [
            params.get("prompt", ""),
            params.get("width", 544),              # Parameter mapping is fine here
            params.get("height", 544),             # Parameter mapping is fine here
            params.get("batch_size", 1),
            params.get("video_length", 25),
            params.get("fps", 24),
            params.get("infer_steps", 30),
            params.get("seed", -1),
            params.get("model", "hunyuan/mp_rank_00_model_states.pt"),
            params.get("vae", "hunyuan/pytorch_model.pt"),
            params.get("te1", "hunyuan/llava_llama3_fp16.safetensors"),
            params.get("te2", "hunyuan/clip_l.safetensors"),
            params.get("save_path", "outputs"),
            params.get("flow_shift", 11.0),
            params.get("cfg_scale", 7.0),
            params.get("output_type", "video"),
            params.get("attn_mode", "sdpa"),
            params.get("block_swap", "0"),
            *[params.get(f"lora{i+1}", "") for i in range(4)],
            *[params.get(f"lora{i+1}_multiplier", 1.0) for i in range(4)]
        ] if params else [gr.update()]*26, # This lambda returns values based on param keys
        inputs=params_state,
        outputs=[prompt, t2v_width, t2v_height, batch_size, video_length, fps, infer_steps, seed, # <<< CORRECTED HERE: use t2v_width, t2v_height
                 model, vae, te1, te2, save_path, flow_shift, cfg_scale,
                 output_type, attn_mode, block_swap] + lora_weights + lora_multipliers
    )
    # Text to Video generation
    generate_btn.click(
        fn=process_batch,
        inputs=[
            prompt, t2v_width, t2v_height, batch_size, video_length, fps, infer_steps,
            seed, dit_folder, model, vae, te1, te2, save_path, flow_shift, cfg_scale,
            output_type, attn_mode, block_swap, exclude_single_blocks, use_split_attn,
            lora_folder, *lora_weights, *lora_multipliers, gr.Textbox(visible=False), gr.Number(visible=False), use_fp8
        ],
        outputs=[video_output, batch_progress, progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[batch_size],
        outputs=selected_index
    )    

    # Update gallery selection handling
    def handle_gallery_select(evt: gr.SelectData) -> int:
        return evt.index

    # Track selected index when gallery item is clicked
    video_output.select(
        fn=handle_gallery_select,
        outputs=selected_index
    )

    # Track selected index when Video2Video gallery item is clicked
    def handle_v2v_gallery_select(evt: gr.SelectData) -> int:
        """Handle gallery selection without automatically updating the input"""
        return evt.index

    # Update the gallery selection event
    v2v_output.select(
        fn=handle_v2v_gallery_select,
        outputs=v2v_selected_index
    )
    
    # Send button handler with gallery selection
    def handle_send_button(
        gallery: list, 
        prompt: str, 
        idx: int, 
        width: int,
        height: int,
        batch_size: int, 
        video_length: int, 
        fps: int, 
        infer_steps: int, 
        seed: int, 
        flow_shift: float, 
        cfg_scale: float,
        lora1: str,
        lora2: str,
        lora3: str,
        lora4: str,
        lora1_multiplier: float,
        lora2_multiplier: float,
        lora3_multiplier: float,
        lora4_multiplier: float
    ) -> tuple:
        if not gallery or idx is None or idx >= len(gallery):
            return (None, "", width, height, batch_size, video_length, fps, infer_steps, 
                    seed, flow_shift, cfg_scale, 
                    lora1, lora2, lora3, lora4,
                    lora1_multiplier, lora2_multiplier, lora3_multiplier, lora4_multiplier,
                    "")  # Add empty string for negative_prompt in the return values

        # Auto-select first item if only one exists and no selection made
        if idx is None and len(gallery) == 1:
            idx = 0

        selected_item = gallery[idx]

        # Handle different gallery item formats
        if isinstance(selected_item, dict):
            video_path = selected_item.get("name", selected_item.get("data", None))
        elif isinstance(selected_item, (tuple, list)):
            video_path = selected_item[0]
        else:
            video_path = selected_item

        # Final cleanup for Gradio Video component
        if isinstance(video_path, tuple):
            video_path = video_path[0]

        return (
            str(video_path), 
            prompt,
            width,
            height, 
            batch_size, 
            video_length, 
            fps, 
            infer_steps, 
            seed, 
            flow_shift, 
            cfg_scale,
            lora1,
            lora2,
            lora3,
            lora4,
            lora1_multiplier,
            lora2_multiplier,
            lora3_multiplier,
            lora4_multiplier,
            ""  # Add empty string for negative_prompt
        )
    
    send_t2v_to_v2v_btn.click(
        fn=handle_send_button,
        inputs=[
            video_output, prompt, selected_index,
            t2v_width, t2v_height, batch_size, video_length,
            fps, infer_steps, seed, flow_shift, cfg_scale
        ] + lora_weights + lora_multipliers,  # Remove the string here
        outputs=[
            v2v_input, 
            v2v_prompt,
            v2v_width,
            v2v_height,
            v2v_batch_size,
            v2v_video_length,
            v2v_fps,
            v2v_infer_steps,
            v2v_seed,
            v2v_flow_shift,
            v2v_cfg_scale
        ] + v2v_lora_weights + v2v_lora_multipliers + [v2v_negative_prompt]
    ).then(
        fn=change_to_tab_two, inputs=None, outputs=[tabs]
    )

    def handle_send_to_v2v(metadata: dict, video_path: str) -> Tuple[str, dict, str]:
        """Handle both parameters and video transfer"""
        status_msg, params = send_parameters_to_tab(metadata, "v2v")
        return status_msg, params, video_path
    
    def handle_info_to_v2v(metadata: dict, video_path: str) -> Tuple[str, Dict, str]:
        """Handle both parameters and video transfer from Video Info to V2V tab"""
        if not video_path:
            return "No video selected", {}, None

        status_msg, params = send_parameters_to_tab(metadata, "v2v")
        # Just return the path directly
        return status_msg, params, video_path

    # Send button click handler
    send_to_v2v_btn.click(
        fn=handle_info_to_v2v,
        inputs=[metadata_output, video_input],
        outputs=[status, params_state, v2v_input]
    ).then(
        lambda params: [
            params.get("v2v_prompt", ""),
            params.get("v2v_width", 544),
            params.get("v2v_height", 544),
            params.get("v2v_batch_size", 1),
            params.get("v2v_video_length", 25),
            params.get("v2v_fps", 24),
            params.get("v2v_infer_steps", 30),
            params.get("v2v_seed", -1),
            params.get("v2v_model", "hunyuan/mp_rank_00_model_states.pt"),
            params.get("v2v_vae", "hunyuan/pytorch_model.pt"),
            params.get("v2v_te1", "hunyuan/llava_llama3_fp16.safetensors"),
            params.get("v2v_te2", "hunyuan/clip_l.safetensors"),
            params.get("v2v_save_path", "outputs"),
            params.get("v2v_flow_shift", 11.0),
            params.get("v2v_cfg_scale", 7.0),
            params.get("v2v_output_type", "video"),
            params.get("v2v_attn_mode", "sdpa"),
            params.get("v2v_block_swap", "0"),
            *[params.get(f"v2v_lora_weights[{i}]", "") for i in range(4)],
            *[params.get(f"v2v_lora_multipliers[{i}]", 1.0) for i in range(4)]
        ] if params else [gr.update()] * 26,
        inputs=params_state,
        outputs=[
            v2v_prompt, v2v_width, v2v_height, v2v_batch_size, v2v_video_length,
            v2v_fps, v2v_infer_steps, v2v_seed, v2v_model, v2v_vae, v2v_te1,
            v2v_te2, v2v_save_path, v2v_flow_shift, v2v_cfg_scale, v2v_output_type,
            v2v_attn_mode, v2v_block_swap
        ] + v2v_lora_weights + v2v_lora_multipliers
    ).then(
        lambda: print(f"Tabs object: {tabs}"),  # Debug print
        outputs=None
    ).then(
        fn=change_to_tab_two, inputs=None, outputs=[tabs]
    )

    # Handler for sending selected video from Video2Video gallery to input
    def handle_v2v_send_button(gallery: list, prompt: str, idx: int) -> Tuple[Optional[str], str]:
        """Send the currently selected video in V2V gallery to V2V input"""
        if not gallery or idx is None or idx >= len(gallery):
            return None, ""

        selected_item = gallery[idx]
        video_path = None

        # Handle different gallery item formats
        if isinstance(selected_item, tuple):
            video_path = selected_item[0]  # Gallery returns (path, caption)
        elif isinstance(selected_item, dict):
            video_path = selected_item.get("name", selected_item.get("data", None))
        elif isinstance(selected_item, str):
            video_path = selected_item

        if not video_path:
            return None, ""

        # Check if the file exists and is accessible
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found at {video_path}")
            return None, ""

        return video_path, prompt

    v2v_send_to_input_btn.click(
        fn=handle_v2v_send_button,
        inputs=[v2v_output, v2v_prompt, v2v_selected_index],
        outputs=[v2v_input, v2v_prompt]
    ).then(
        lambda: gr.update(visible=True),  # Ensure the video input is visible
        outputs=v2v_input
    )

    # Video to Video generation
    v2v_generate_btn.click(
        fn=process_batch,
        inputs=[
            v2v_prompt, v2v_width, v2v_height, v2v_batch_size, v2v_video_length, 
            v2v_fps, v2v_infer_steps, v2v_seed, v2v_dit_folder, v2v_model, v2v_vae, v2v_te1, v2v_te2,
            v2v_save_path, v2v_flow_shift, v2v_cfg_scale, v2v_output_type, v2v_attn_mode, 
            v2v_block_swap, v2v_exclude_single_blocks, v2v_use_split_attn, v2v_lora_folder, 
            *v2v_lora_weights, *v2v_lora_multipliers, v2v_input, v2v_strength,
            v2v_negative_prompt, v2v_cfg_scale, v2v_split_uncond, v2v_use_fp8
        ],
        outputs=[v2v_output, v2v_batch_progress, v2v_progress_text],
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[v2v_batch_size],
        outputs=v2v_selected_index
    )
    refresh_outputs = [model]  # Add model dropdown to outputs
    for i in range(4):
        refresh_outputs.extend([lora_weights[i], lora_multipliers[i]])
    
    refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,
        inputs=[dit_folder, lora_folder, model] + lora_weights + lora_multipliers,
        outputs=refresh_outputs
    )
    # Image2Video refresh
    i2v_refresh_outputs = [i2v_model]  # Add model dropdown to outputs
    for i in range(4):
        i2v_refresh_outputs.extend([i2v_lora_weights[i], i2v_lora_multipliers[i]])
    
    i2v_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,
        inputs=[i2v_dit_folder, i2v_lora_folder, i2v_model] + i2v_lora_weights + i2v_lora_multipliers,
        outputs=i2v_refresh_outputs
    )
    
    # Video2Video refresh
    v2v_refresh_outputs = [v2v_model]  # Add model dropdown to outputs
    for i in range(4):
        v2v_refresh_outputs.extend([v2v_lora_weights[i], v2v_lora_multipliers[i]])
    
    v2v_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,
        inputs=[v2v_dit_folder, v2v_lora_folder, v2v_model] + v2v_lora_weights + v2v_lora_multipliers,
        outputs=v2v_refresh_outputs
    )

    # WanX-i2v tab connections
    wanx_prompt.change(fn=count_prompt_tokens, inputs=wanx_prompt, outputs=wanx_token_counter)
    wanx_stop_btn.click(fn=lambda: stop_event.set(), queue=False)
    
    # Image input handling for WanX-i2v
    wanx_input.change(
        fn=update_wanx_image_dimensions,
        inputs=[wanx_input],
        outputs=[wanx_original_dims, wanx_width, wanx_height]
    )

    # Scale slider handling for WanX-i2v
    wanx_scale_slider.change(
        fn=update_wanx_from_scale,
        inputs=[wanx_scale_slider, wanx_original_dims],
        outputs=[wanx_width, wanx_height]
    )

    # Width/height calculation buttons for WanX-i2v
    wanx_calc_width_btn.click(
        fn=calculate_wanx_width,
        inputs=[wanx_height, wanx_original_dims],
        outputs=[wanx_width]
    )

    wanx_calc_height_btn.click(
        fn=calculate_wanx_height,
        inputs=[wanx_width, wanx_original_dims],
        outputs=[wanx_height]
    )
    # Add visibility toggle for the folder input components
    wanx_use_random_folder.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x), gr.update(visible=x), gr.update(visible=not x)),
        inputs=[wanx_use_random_folder],
        outputs=[wanx_input_folder, wanx_folder_status, wanx_validate_folder_btn, wanx_input]
    )
    def toggle_end_image(use_end_image):
        return (
            gr.update(visible=use_end_image, interactive=use_end_image),  # wanx_input_end
            gr.update(visible=False)  # wanx_trim_frames
        )
    wanx_use_end_image.change(
        fn=toggle_end_image,
        inputs=[wanx_use_end_image],
        outputs=[wanx_input_end, wanx_trim_frames]
    )
    # Validate folder button handler
    wanx_validate_folder_btn.click(
        fn=lambda folder: get_random_image_from_folder(folder)[1],
        inputs=[wanx_input_folder],
        outputs=[wanx_folder_status]
    )

    # Flow shift recommendation buttons
    wanx_recommend_flow_btn.click(
        fn=recommend_wanx_flow_shift,
        inputs=[wanx_width, wanx_height],
        outputs=[wanx_flow_shift]
    )

    wanx_t2v_recommend_flow_btn.click(
        fn=recommend_wanx_flow_shift,
        inputs=[wanx_t2v_width, wanx_t2v_height],
        outputs=[wanx_t2v_flow_shift]
    )
    
    # Generate button handler
    wanx_generate_btn.click(
        fn=wanx_batch_handler,
        inputs=[
            wanx_use_random_folder,
            wanx_prompt,
            wanx_negative_prompt,
            wanx_width,
            wanx_height,
            wanx_video_length,
            wanx_fps,
            wanx_infer_steps,
            wanx_flow_shift,
            wanx_guidance_scale,
            wanx_seed,
            wanx_batch_size,
            wanx_input_folder,
            wanx_input_end, # Make sure this is passed
            wanx_task,
            wanx_dit_folder,
            wanx_dit_path,
            wanx_vae_path,
            wanx_t5_path,
            wanx_clip_path,
            wanx_save_path,
            wanx_output_type,
            wanx_sample_solver,
            wanx_exclude_single_blocks,
            wanx_attn_mode,
            wanx_block_swap,
            wanx_fp8,
            wanx_fp8_scaled,
            wanx_fp8_t5,
            wanx_lora_folder,
            wanx_slg_layers,
            wanx_slg_start,
            wanx_slg_end,
            wanx_enable_cfg_skip,
            wanx_cfg_skip_mode,
            wanx_cfg_apply_ratio,
            # --- ADDED PREVIEW INPUTS ---
            wanx_enable_preview,
            wanx_preview_steps,
            # --- END ADDED ---
            *wanx_lora_weights,
            *wanx_lora_multipliers,
            wanx_input,              # Input image (used as input_file in handler)
            wanx_control_video,      # Control video
            wanx_control_strength,
            wanx_control_start,
            wanx_control_end,
        ],
        outputs=[
            wanx_output,          # Main video gallery
            wanx_preview_output,  # ADDED: Preview gallery
            wanx_batch_progress,  # Status text
            wanx_progress_text    # Progress text
        ], # Now 4 outputs
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[wanx_batch_size],
        outputs=wanx_i2v_selected_index
    )
    
    # Add refresh button handler for WanX-i2v tab
    wanx_refresh_outputs = [wanx_dit_path]  # Add model dropdown to outputs
    for i in range(4):
        wanx_refresh_outputs.extend([wanx_lora_weights[i], wanx_lora_multipliers[i]])

    wanx_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,  # This function already exists and handles both updates
        inputs=[wanx_dit_folder, wanx_lora_folder, wanx_dit_path] + wanx_lora_weights + wanx_lora_multipliers,
        outputs=wanx_refresh_outputs
    )
    wanx_dit_folder.change(
        fn=update_dit_dropdown,
        inputs=[wanx_dit_folder],
        outputs=[wanx_dit_path]
    )

    wanx_dit_folder.change(
        fn=update_dit_dropdown,
        inputs=[wanx_dit_folder],
        outputs=[wanx_t2v_dit_path]
    )

    wanx_dit_folder.change(
        fn=update_dit_dropdown,
        inputs=[wanx_dit_folder],
        outputs=[wanx_v2v_dit_path]
    )
    
    # Gallery selection handling
    wanx_output.select(
        fn=handle_wanx_gallery_select,
        inputs=[wanx_output],
        outputs=[wanx_i2v_selected_index, wanx_base_video]
    )
    
    # Send to Video2Video handler
    wanx_send_to_v2v_btn.click(
        fn=send_wanx_to_v2v,
        inputs=[
            wanx_output,  # Gallery with videos
            wanx_prompt,  # Prompt text
            wanx_i2v_selected_index,  # Use the correct selected index state
            wanx_width, 
            wanx_height, 
            wanx_video_length,
            wanx_fps, 
            wanx_infer_steps, 
            wanx_seed,
            wanx_flow_shift, 
            wanx_guidance_scale,
            wanx_negative_prompt
        ],
        outputs=[
            v2v_input,  # Video input in V2V tab
            v2v_prompt,  # Prompt in V2V tab
            v2v_width, 
            v2v_height,
            v2v_video_length, 
            v2v_fps, 
            v2v_infer_steps,
            v2v_seed, 
            v2v_flow_shift, 
            v2v_cfg_scale,
            v2v_negative_prompt
        ]
    ).then(
        fn=change_to_tab_two,  # Function to switch to Video2Video tab
        inputs=None,
        outputs=[tabs]
    )
    # Connect prompt token counter
    wanx_t2v_prompt.change(fn=count_prompt_tokens, inputs=wanx_t2v_prompt, outputs=wanx_t2v_token_counter)

    # Stop button handler
    wanx_t2v_stop_btn.click(fn=lambda: stop_event.set(), queue=False)

    # Flow shift recommendation button
    wanx_t2v_recommend_flow_btn.click(
        fn=recommend_wanx_flow_shift,
        inputs=[wanx_t2v_width, wanx_t2v_height],
        outputs=[wanx_t2v_flow_shift]
    )

    # Task change handler to update CLIP visibility and path
    def update_clip_visibility(task):
        is_i2v = "i2v" in task
        return gr.update(visible=is_i2v)

    wanx_t2v_task.change(
        fn=update_clip_visibility,
        inputs=[wanx_t2v_task],
        outputs=[wanx_t2v_clip_path]
    )

        # Generate button handler for T2V
    wanx_t2v_generate_btn.click(
        fn=wanx_batch_handler,
        inputs=[
            wanx_t2v_use_random_folder, # use_random
            wanx_t2v_prompt,            # prompt
            wanx_t2v_negative_prompt,   # negative_prompt
            wanx_t2v_width,             # width
            wanx_t2v_height,            # height
            wanx_t2v_video_length,      # video_length
            wanx_t2v_fps,               # fps
            wanx_t2v_infer_steps,       # infer_steps
            wanx_t2v_flow_shift,        # flow_shift
            wanx_t2v_guidance_scale,    # guidance_scale
            wanx_t2v_seed,              # seed
            wanx_t2v_batch_size,        # batch_size
            wanx_t2v_input_folder,      # input_folder_path
            wanx_t2v_input_end,         # wanx_input_end
            wanx_t2v_task,              # task
            wanx_dit_folder,            # dit_folder (shared)
            wanx_t2v_dit_path,          # dit_path
            wanx_t2v_vae_path,          # vae_path
            wanx_t2v_t5_path,           # t5_path
            wanx_t2v_clip_path,         # clip_path (often None for t2v)
            wanx_t2v_save_path,         # save_path
            wanx_t2v_output_type,       # output_type
            wanx_t2v_sample_solver,     # sample_solver
            wanx_t2v_exclude_single_blocks, # exclude_single_blocks
            wanx_t2v_attn_mode,         # attn_mode
            wanx_t2v_block_swap,        # block_swap
            wanx_t2v_fp8,               # fp8
            wanx_t2v_fp8_scaled,        # fp8_scaled
            wanx_t2v_fp8_t5,            # fp8_t5
            wanx_t2v_lora_folder,       # lora_folder
            wanx_t2v_slg_layers,        # slg_layers
            wanx_t2v_slg_start,         # slg_start
            wanx_t2v_slg_end,           # slg_end
            wanx_t2v_enable_cfg_skip,   # enable_cfg_skip
            wanx_t2v_cfg_skip_mode,     # cfg_skip_mode
            wanx_t2v_cfg_apply_ratio,   # cfg_apply_ratio
            # --- ADDED PREVIEW INPUTS ---
            wanx_t2v_enable_preview,
            wanx_t2v_preview_steps,
            # --- END ADDED ---
            *wanx_t2v_lora_weights,     # *lora_params (weights)
            *wanx_t2v_lora_multipliers, # *lora_params (multipliers)
            # --- ADDED Placeholders for trailing args expected by wanx_batch_handler ---
            gr.File(value=None, visible=False), # Placeholder for input_file (None for T2V)
            gr.Video(value=None, visible=False), # Placeholder for control_video (None for T2V)
            gr.Number(value=1.0, visible=False), # Placeholder for control_strength
            gr.Number(value=0.0, visible=False), # Placeholder for control_start
            gr.Number(value=1.0, visible=False), # Placeholder for control_end
            # --- END Placeholders ---
        ],
        outputs=[
            wanx_t2v_output,         # Main video gallery
            wanx_t2v_preview_output, # ADDED: Preview gallery
            wanx_t2v_batch_progress, # Status text
            wanx_t2v_progress_text   # Progress text
        ], # Now 4 outputs
        queue=True
    ).then(
        fn=lambda batch_size: 0 if batch_size == 1 else None,
        inputs=[wanx_t2v_batch_size],
        outputs=wanx_t2v_selected_index
    )
    
    # Add refresh button handler for WanX-t2v tab
    wanx_t2v_refresh_outputs = [wanx_t2v_dit_path]  # This is one output
    for i in range(4):
        wanx_t2v_refresh_outputs.extend([wanx_t2v_lora_weights[i], wanx_t2v_lora_multipliers[i]])  # This adds 8 more outputs

    wanx_t2v_refresh_btn.click(
        fn=update_dit_and_lora_dropdowns,  # Change to this function instead
        inputs=[wanx_dit_folder, wanx_t2v_lora_folder, wanx_t2v_dit_path] + wanx_t2v_lora_weights + wanx_t2v_lora_multipliers,
        outputs=wanx_t2v_refresh_outputs
    )

    # Gallery selection handling
    wanx_t2v_output.select(
        fn=handle_wanx_t2v_gallery_select,
        outputs=wanx_t2v_selected_index
    )

    # Send to Video2Video handler
    wanx_t2v_send_to_v2v_btn.click(
        fn=send_wanx_t2v_to_v2v,
        inputs=[
            wanx_t2v_output, 
            wanx_t2v_prompt, 
            wanx_t2v_selected_index,
            wanx_t2v_width, 
            wanx_t2v_height, 
            wanx_t2v_video_length,
            wanx_t2v_fps, 
            wanx_t2v_infer_steps, 
            wanx_t2v_seed,
            wanx_t2v_flow_shift, 
            wanx_t2v_guidance_scale,
            wanx_t2v_negative_prompt
        ],
        outputs=[
            v2v_input, 
            v2v_prompt, 
            v2v_width, 
            v2v_height,
            v2v_video_length, 
            v2v_fps, 
            v2v_infer_steps,
            v2v_seed, 
            v2v_flow_shift, 
            v2v_cfg_scale,
            v2v_negative_prompt
        ]
    ).then(
        fn=change_to_tab_two,
        inputs=None,
        outputs=[tabs]
    )
if __name__ == "__main__":
    # Make sure 'outputs' directory exists
    os.makedirs("outputs", exist_ok=True)
    # Optional: Clean temp_frames directory on startup
    #if os.path.exists("temp_frames"):
    #    try: shutil.rmtree("temp_frames")
    #    except OSError as e: print(f"Error removing temp_frames: {e}")
    os.makedirs("temp_frames", exist_ok=True)

demo.queue().launch(server_name="0.0.0.0", share=False)