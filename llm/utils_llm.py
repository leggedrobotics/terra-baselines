import base64
import cv2
import numpy as np
import jax
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
import jax.numpy as jnp
from llm.adk_llm import *

import csv
from utils.models import load_neural_network

import json
import jax.numpy as jnp
import ast
import io
from PIL import Image
import matplotlib.pyplot as plt

import jax.numpy as jnp
import pygame as pg

from llm.session_manager_llm import SessionManager
from llm.prompt_manager_llm import PromptManager 
import os
import yaml
import datetime
import random

def encode_image(cv_image):
    _, buffer = cv2.imencode(".jpg", cv_image)
    return base64.b64encode(buffer).decode("utf-8")

def save_csv(output_file, action_list, cumulative_rewards):
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["actions", "cumulative_rewards"]) # Header updated
        # Iterate through actions and the calculated cumulative rewards
        for action, cum_reward in zip(action_list, cumulative_rewards):
            # Assuming action is array-like (e.g., JAX array) with one element
            action_value = action[0] if hasattr(action, '__getitem__') and len(action) > 0 else action
            # cum_reward from np.cumsum is already a scalar number
            reward_value = cum_reward
            writer.writerow([action_value, reward_value])

    print(f"Results saved to {output_file}")

def create_sub_task_target_map_64x64(global_target_map_data: jnp.ndarray,
                                     region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    Creates a 64x64 target map for an RL agent's sub-task from a 64x64 input.
    
    Retains both `-1` values (dig targets) and `1` values (dump targets) from 
    the specified region in the global map. Everything outside the region is set to 0 (free).
    
    Args:
        global_target_map_data: Target map of size 64x64 (1: dump, 0: free, -1: dig).
        region_coords: (y_start, x_start, y_end, x_end), inclusive bounds.
    
    Returns:
        A new 64x64 map with `-1`s and `1`s from the region at their original positions;
        everything else is 0.
    """
    y_start, x_start, y_end, x_end = region_coords
    
    # Initialize a 64x64 map with all zeros (free space)
    #sub_task_map = jnp.zeros((64, 64), dtype=global_target_map_data.dtype)
    # Initialize a 64x64 map with all ones (dump space)
    sub_task_map = jnp.ones((64, 64), dtype=global_target_map_data.dtype)
    
    # Define slice object for region
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    
    # Extract region from global map
    region_data = global_target_map_data[region_slice]
    
    # Place region data at the SAME position in the new map
    sub_task_map = sub_task_map.at[region_slice].set(region_data)
    
    return sub_task_map

def create_sub_task_target_map_64x64_big(global_target_map_data: jnp.ndarray,
                                           region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    FIXED: Creates a 64x64 target map that always returns exactly 64x64 dimensions.
    
    Args:
        global_target_map_data: Target map of any size (1: dump, 0: free, -1: dig).
        region_coords: (y_start, x_start, y_end, x_end), inclusive bounds.
    
    Returns:
        A 64x64 map with the extracted region data placed appropriately.
    """
    y_start, x_start, y_end, x_end = region_coords
    
    # Always initialize a 64x64 map
    sub_task_map = jnp.ones((64, 64), dtype=global_target_map_data.dtype)
    
    # Calculate the actual region size
    region_height = y_end - y_start + 1
    region_width = x_end - x_start + 1
    
    # Extract the region from global map
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    region_data = global_target_map_data[region_slice]
    
    # Ensure region_data is exactly the expected size
    if region_data.shape != (region_height, region_width):
        print(f"Warning: Region data shape {region_data.shape} doesn't match expected {(region_height, region_width)}")
        # Crop or pad as needed
        min_h = min(region_data.shape[0], region_height)
        min_w = min(region_data.shape[1], region_width)
        region_data = region_data[:min_h, :min_w]
    
    # Calculate how much of the region we can fit in 64x64
    fit_height = min(region_height, 64)
    fit_width = min(region_width, 64)
    
    # Place the region data in the 64x64 map
    # Option 1: Place at origin (0,0)
    sub_task_map = sub_task_map.at[:fit_height, :fit_width].set(region_data[:fit_height, :fit_width])
    
    # Option 2: Place at the same relative position (if it fits)
    # if region_height <= 64 and region_width <= 64:
    #     # Calculate offset to maintain relative position
    #     offset_y = min(y_start, 64 - region_height)
    #     offset_x = min(x_start, 64 - region_width)
    #     sub_task_map = sub_task_map.at[offset_y:offset_y+region_height, offset_x:offset_x+region_width].set(region_data)
    
    return sub_task_map

def create_sub_task_action_map_64x64_big(action_map_data: jnp.ndarray,
                                          region_coords: tuple[int, int, int, int]) -> jnp.ndarray:
    """
    FIXED: Creates a 64x64 action map that always returns exactly 64x64 dimensions.
    """
    y_start, x_start, y_end, x_end = region_coords
    
    # Always initialize a 64x64 map
    sub_task_map = jnp.zeros((64, 64), dtype=action_map_data.dtype)
    
    # Calculate the actual region size
    region_height = y_end - y_start + 1
    region_width = x_end - x_start + 1
    
    # Extract the region from global map
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    region_data = action_map_data[region_slice]
    
    # Calculate how much of the region we can fit in 64x64
    fit_height = min(region_height, 64)
    fit_width = min(region_width, 64)
    
    # Place the region data in the 64x64 map
    sub_task_map = sub_task_map.at[:fit_height, :fit_width].set(region_data[:fit_height, :fit_width])
    
    return sub_task_map

def create_sub_task_mask_64x64_big(mask_data: jnp.ndarray,
                                    region_coords: tuple[int, int, int, int],
                                    default_value: int = 1) -> jnp.ndarray:
    """
    FIXED: Generic function to create 64x64 masks that always return exactly 64x64 dimensions.
    
    Args:
        mask_data: Input mask of any size
        region_coords: (y_start, x_start, y_end, x_end), inclusive bounds
        default_value: Default value for areas outside the region (1 for non-traversable, 0 for traversable)
    """
    y_start, x_start, y_end, x_end = region_coords
    
    # Always initialize a 64x64 map
    sub_task_map = jnp.full((64, 64), default_value, dtype=mask_data.dtype)
    
    # Calculate the actual region size
    region_height = y_end - y_start + 1
    region_width = x_end - x_start + 1
    
    # Extract the region from global map
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    region_data = mask_data[region_slice]
    
    # Calculate how much of the region we can fit in 64x64
    fit_height = min(region_height, 64)
    fit_width = min(region_width, 64)
    
    # Place the region data in the 64x64 map
    sub_task_map = sub_task_map.at[:fit_height, :fit_width].set(region_data[:fit_height, :fit_width])
    
    return sub_task_map

# Wrapper functions for specific mask types
def create_sub_task_padding_mask_64x64_big(padding_mask_data, region_coords):
    return create_sub_task_mask_64x64_big(padding_mask_data, region_coords, default_value=1)

def create_sub_task_traversability_mask_64x64_big(traversability_mask_data, region_coords):
    return create_sub_task_mask_64x64_big(traversability_mask_data, region_coords, default_value=1)

def create_sub_task_dumpability_mask_64x64_big(dumpability_mask_data, region_coords):
    return create_sub_task_mask_64x64_big(dumpability_mask_data, region_coords, default_value=0)

def extract_python_format_data(llm_response_text):
    """
    Extracts Python-formatted data from LLM response, preserving tuples.
    
    Args:
        llm_response_text (str): The raw text response from the LLM
        
    Returns:
        list: The parsed Python list with tuples preserved
        
    Raises:
        ValueError: If no valid Python data could be extracted
    """
    # First, check if we have a code block and extract its content
    code_block_pattern = r'```(?:json|python)?\s*([\s\S]*?)\s*```'
    code_match = re.search(code_block_pattern, llm_response_text, re.DOTALL)
    
    if code_match:
        content = code_match.group(1)
    else:
        # If no code block, use the whole text
        content = llm_response_text
    
    # Clean up the content to ensure it's valid Python syntax
    # Replace double quotes with single quotes for keys (Python style)
    content = re.sub(r'"([^"]+)":', r"'\1':", content)
    
    # Make sure status values are properly quoted
    content = re.sub(r"'status':\s*([a-zA-Z_][a-zA-Z0-9_]*)", r"'status': '\1'", content)
    
    try:
        # Use ast.literal_eval to parse the Python literals, which preserves tuples
        return ast.literal_eval(content)
    except (SyntaxError, ValueError) as e:
        logger.warning(f"ast.literal_eval failed: {e}")
        
        # Try to extract and process each dict individually
        results = []
        dict_pattern = r'\{\s*\'id\':\s*(\d+)[\s\S]*?(?=\}\s*,|\}\s*$)'
        
        for match in re.finditer(dict_pattern, content, re.DOTALL):
            try:
                dict_str = match.group(0) + '}'
                # Make sure all string values are properly quoted
                dict_str = re.sub(r"'([^']+)':\s*([a-zA-Z_][a-zA-Z0-9_]*)", r"'\1': '\2'", dict_str)
                obj = ast.literal_eval(dict_str)
                results.append(obj)
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Failed to parse dict: {e}")
                continue
        
        if results:
            return results
    
    # If we still couldn't parse it, try a more manual approach
    try:
        # Extract data manually using regex
        result = []
        id_pattern = r"'id':\s*(\d+)"
        region_pattern = r"'region_coords':\s*\(([^)]+)\)"
        pos_pattern = r"'start_pos':\s*\(([^)]+)\)"
        angle_pattern = r"'start_angle':\s*(\d+)"
        status_pattern = r"'status':\s*'([^']+)'"
        
        # Get all IDs
        ids = re.findall(id_pattern, content)
        region_coords = re.findall(region_pattern, content)
        start_positions = re.findall(pos_pattern, content)
        start_angles = re.findall(angle_pattern, content)
        statuses = re.findall(status_pattern, content)
        
        # Ensure we have the same number of matches for each field
        min_length = min(len(ids), len(region_coords), len(start_positions), 
                         len(start_angles), len(statuses))
        
        for i in range(min_length):
            # Parse tuple values
            region_tuple = tuple(int(x.strip()) for x in region_coords[i].split(','))
            start_pos_tuple = tuple(int(x.strip()) for x in start_positions[i].split(','))
            
            obj = {
                'id': int(ids[i]),
                'region_coords': region_tuple,
                'start_pos': start_pos_tuple,
                'start_angle': int(start_angles[i]),
                'status': statuses[i]
            }
            result.append(obj)
        
        if result:
            return result
    except Exception as e:
        logger.error(f"Manual extraction failed: {e}")
    
    raise ValueError("Could not extract valid Python data with tuples from LLM response")

def is_valid_region_list(var):
    """
    Checks if the variable is a list of dictionaries with the required structure.
    The structure should be a list containing at least one dictionary with the keys:
    'id', 'region_coords', 'start_pos', 'start_angle', and 'status'.
    
    'region_coords' and 'start_pos' should be tuples.
    Each region must be maximum 64x64 in size.
    
    Example of valid structure:
    [{'id': 0, 'region_coords': (15, 15, 50, 50), 'start_pos': (42, 36), 'start_angle': 0, 'status': 'pending'}]
    
    Args:
        var: The variable to check
        
    Returns:
        bool: True if the variable has the valid structure, False otherwise
    """
    # Check if var is a list
    if not isinstance(var, list):
        return False
    
    # Check if list has at least one element
    if len(var) == 0:
        return False
    
    # Maximum allowed partition size
    MAX_PARTITION_SIZE = 64
    
    # Check each element in the list
    for item in var:
        # Check if item is a dictionary
        if not isinstance(item, dict):
            return False
        
        # Check required keys
        required_keys = {'id', 'region_coords', 'start_pos', 'start_angle', 'status'}
        if set(item.keys()) != required_keys:
            return False
        
        # Check types of specific fields
        if not isinstance(item['id'], (int, float)):
            return False
        
        if not isinstance(item['region_coords'], tuple) or len(item['region_coords']) != 4:
            return False
            
        if not isinstance(item['start_pos'], tuple) or len(item['start_pos']) != 2:
            return False
            
        if not isinstance(item['start_angle'], (int, float)):
            return False
            
        if not isinstance(item['status'], str):
            return False
        
        # Check partition size (region_coords format: (y_start, x_start, y_end, x_end))
        y_start, x_start, y_end, x_end = item['region_coords']
        if not all(isinstance(coord, (int, float)) for coord in item['region_coords']):
            return False
        
        # Calculate width and height from coordinates
        width = x_end - x_start
        height = y_end - y_start

        print(f"Checking partition: {item['id']} with width {width} and height {height}")
        
        if width > MAX_PARTITION_SIZE or height > MAX_PARTITION_SIZE:
            return False
    
    return True

def compute_manual_subtasks(ORIGINAL_MAP_SIZE, NUM_PARTITIONS):
    """
    UPDATED: Ensures all partitions are exactly 64x64 for RL policy compatibility.
    """
    if ORIGINAL_MAP_SIZE not in [64, 128]:
        raise ValueError(f"Unsupported ORIGINAL_MAP_SIZE: {ORIGINAL_MAP_SIZE}. Must be 64 or 128.")
    
    if NUM_PARTITIONS == 1:
        if ORIGINAL_MAP_SIZE == 64:
            sub_tasks_manual = [
                {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
            ]
        else:  # 128x128
            # Take a 64x64 region from the 128x128 map
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
            # ]

            # Single partition for 128x128 map, taking the center 64x64 region
            sub_tasks_manual = [
                {'id': 0, 'region_coords': (32, 32, 95, 95), 'start_pos': (64, 64), 'start_angle': 0, 'status': 'pending'},
            ]


    elif NUM_PARTITIONS == 2:
        if ORIGINAL_MAP_SIZE == 64:
            # Horizontal split of 64x64 map
            sub_tasks_manual = [
                {'id': 0, 'region_coords': (0, 0, 31, 63), 'start_pos': (16, 32), 'start_angle': 0, 'status': 'pending'},
                {'id': 1, 'region_coords': (32, 0, 63, 63), 'start_pos': (48, 32), 'start_angle': 0, 'status': 'pending'}
            ]

            # Horizontal partitioning for 64x64 map (x,y)
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (0, 0, 31, 63), 'start_pos': (32, 16), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 1, 'region_coords': (32, 0, 63, 63), 'start_pos': (32, 48), 'start_angle': 0, 'status': 'pending'}
            # ]

            # Vertical partitioning for 64x64 map
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (0, 0, 63, 31), 'start_pos': (32, 16), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 1, 'region_coords': (0, 32, 63, 63), 'start_pos': (32, 48), 'start_angle': 0, 'status': 'pending'}
            # ]

            # Vertical partitioning with overlapping
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (0, 0, 63, 35), 'start_pos': (32, 18), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 1, 'region_coords': (0, 28, 63, 63), 'start_pos': (32, 46), 'start_angle': 0, 'status': 'pending'}
            # ]

            # Random partitioning for 64x64 map
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (0, 0, 32, 32), 'start_pos': (25, 20), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 1, 'region_coords': (0, 33, 63, 63), 'start_pos': (40, 40), 'start_angle': 0, 'status': 'pending'},
            # ]

        else:  # 128x128
            # Two 64x64 regions from different parts of the 128x128 map
            sub_tasks_manual = [
                {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
                {'id': 1, 'region_coords': (64, 0, 127, 63), 'start_pos': (96, 32), 'start_angle': 0, 'status': 'pending'}
            ]
            
            # Two overlapping 64x64 regions in the center of the 128x128 map
            # Left-center region: (16, 16) to (79, 79)
            # Right-center region: (48, 48) to (111, 111)
            # Overlap area: (48, 48) to (79, 79) = 32x32 overlap
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (16, 16, 79, 79), 'start_pos': (48, 48), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 1, 'region_coords': (48, 48, 111, 111), 'start_pos': (80, 80), 'start_angle': 0, 'status': 'pending'}
            # ]

            # Two overlapping 64x64 regions in the center of the 128x128 map shifting the start positions
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (32, 0, 95, 63), 'start_pos': (64, 32), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 1, 'region_coords': (32, 64, 95, 127), 'start_pos': (64, 96), 'start_angle': 0, 'status': 'pending'}
            # ]
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (32, 0, 95, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 1, 'region_coords': (32, 48, 95, 111), 'start_pos': (32, 80), 'start_angle': 0, 'status': 'pending'}
            # ]
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (32, 0, 95, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 1, 'region_coords': (32, 54, 95, 117), 'start_pos': (32, 80), 'start_angle': 0, 'status': 'pending'}
            # ]
            
    elif NUM_PARTITIONS == 4:
        if ORIGINAL_MAP_SIZE == 64:
            # 2x2 grid of 32x32 regions (these will be padded to 64x64)
            sub_tasks_manual = [
                {'id': 0, 'region_coords': (0, 0, 31, 31), 'start_pos': (16, 16), 'start_angle': 0, 'status': 'pending'},
                {'id': 1, 'region_coords': (0, 32, 31, 63), 'start_pos': (16, 48), 'start_angle': 0, 'status': 'pending'},
                {'id': 2, 'region_coords': (32, 0, 63, 31), 'start_pos': (48, 16), 'start_angle': 0, 'status': 'pending'},
                {'id': 3, 'region_coords': (32, 32, 63, 63), 'start_pos': (48, 48), 'start_angle': 0, 'status': 'pending'}
            ]
        else:  # 128x128
            # Four 64x64 regions covering different corners of the 128x128 map
            sub_tasks_manual = [
                {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (32, 32), 'start_angle': 0, 'status': 'pending'},
                {'id': 1, 'region_coords': (0, 64, 63, 127), 'start_pos': (32, 96), 'start_angle': 0, 'status': 'pending'},
                {'id': 2, 'region_coords': (64, 0, 127, 63), 'start_pos': (96, 32), 'start_angle': 0, 'status': 'pending'},
                {'id': 3, 'region_coords': (64, 64, 127, 127), 'start_pos': (96, 96), 'start_angle': 0, 'status': 'pending'}
            ]
            # Four 64x64 regions covering different corners of the 128x128 map different start positions
            # sub_tasks_manual = [
            #     {'id': 0, 'region_coords': (0, 0, 63, 63), 'start_pos': (20, 20), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 1, 'region_coords': (0, 64, 63, 127), 'start_pos': (20, 44), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 2, 'region_coords': (64, 0, 127, 63), 'start_pos': (44, 20), 'start_angle': 0, 'status': 'pending'},
            #     {'id': 3, 'region_coords': (64, 64, 127, 127), 'start_pos': (44, 44), 'start_angle': 0, 'status': 'pending'}
            # ] 
    else:
        raise ValueError("Invalid number of partitions. Must be 1, 2 or 4.")
    
    # Validate all partitions
    for partition in sub_tasks_manual:
        region_coords = partition['region_coords']
        y_start, x_start, y_end, x_end = region_coords
        height = y_end - y_start + 1
        width = x_end - x_start + 1
        print(f"Partition {partition['id']}: {height}x{width} (will be processed as 64x64)")
        
        # Ensure we don't exceed the original map boundaries
        if y_end >= ORIGINAL_MAP_SIZE or x_end >= ORIGINAL_MAP_SIZE:
            raise ValueError(f"Partition {partition['id']} exceeds map boundaries")
    
    return sub_tasks_manual

def check_overall_completion(partition_states):
    """
    Check if the overall task is complete based on partition completion status.
    Returns True if all partitions are completed or if sufficient progress has been made.
    """
    if not partition_states:
        return False
    
    completed_partitions = []
    failed_partitions = []
    active_partitions = []
    
    for partition_idx, partition_state in partition_states.items():
        status = partition_state.get('status', 'unknown')
        if status == 'completed':
            completed_partitions.append(partition_idx)
        elif status == 'failed':
            failed_partitions.append(partition_idx)
        elif status == 'active':
            active_partitions.append(partition_idx)
    
    total_partitions = len(partition_states)

    
    all_completed = len(completed_partitions) == total_partitions

    is_complete = all_completed
    
    return is_complete

def calculate_map_completion_metrics(partition_states):
    """
    Calculate completion metrics for the current map based on partition states.
    """
    if not partition_states:
        return {
            'done': False,
            'completion_rate': 0.0,
            'total_reward': 0.0,
            'completed_partitions': 0,
            'failed_partitions': 0,
            'total_partitions': 0
        }
    
    completed_count = 0
    failed_count = 0
    total_reward = 0.0
    total_partitions = len(partition_states)
    
    for partition_idx, partition_state in partition_states.items():
        status = partition_state.get('status', 'unknown')
        partition_reward = partition_state.get('total_reward', 0.0)
        total_reward += partition_reward
        
        if status == 'completed':
            completed_count += 1
        elif status == 'failed':
            failed_count += 1
    
    completion_rate = completed_count / total_partitions if total_partitions > 0 else 0.0
    is_done = check_overall_completion(partition_states)
    
    return {
        'done': is_done,
        'completion_rate': completion_rate,
        'total_reward': total_reward,
        'completed_partitions': completed_count,
        'failed_partitions': failed_count,
        'total_partitions': total_partitions
    }

def wrap_action_llm(action_rl, action_type):
    """
    Wrap RL action for the environment.
    Ensures correct shape for single environment (non-batched).
    """
    # Ensure action_rl is a single integer, not an array
    if isinstance(action_rl, jnp.ndarray):
        if action_rl.shape == (1,):
            action_val = action_rl[0]  # Extract single value
        elif action_rl.shape == ():
            action_val = action_rl  # Already scalar
        else:
            raise ValueError(f"Unexpected action shape: {action_rl.shape}")
    else:
        action_val = action_rl
    
    # Convert to proper format for single environment
    # Shape should be [1] not [1,1]
    wrapped_action = action_type(
        type=jnp.array([action_val], dtype=jnp.int8),  # Shape: [1]
        action=jnp.array([action_val], dtype=jnp.int8)  # Shape: [1]
    )

    return wrapped_action

def add_batch_dimension_to_observation(obs):
    """Add batch dimension to all observation components."""
    batched_obs = {}
    for key, value in obs.items():
        if isinstance(value, jnp.ndarray):
            batched_obs[key] = jnp.expand_dims(value, axis=0)
        else:
            batched_obs[key] = jnp.array([value])
    return batched_obs

def reset_to_next_map(map_index, seed, env_manager, global_env_config,
                       initial_custom_pos=None, initial_custom_angle=None):
    """Reset the existing environment to the next map"""
    print(f"\n{'='*60}")
    print(f"RESETTING TO MAP {map_index}")
    print(f"{'='*60}")
        
    # Create new seed for this map reset
    map_seed = seed + map_index * 1000
    map_rng = jax.random.PRNGKey(map_seed)
    map_rng, reset_rng = jax.random.split(map_rng)
    reset_keys = jax.random.split(reset_rng, 1)

    # Reset the existing environment to get a new map
    # The environment will internally cycle through its available maps
    env_manager.global_env.timestep = env_manager.global_env.reset(
        global_env_config, reset_keys, initial_custom_pos, initial_custom_angle
    )

    # Extract and store the NEW global map data directly
    new_timestep = env_manager.global_env.timestep
    env_manager.global_maps['target_map'] = new_timestep.state.world.target_map.map[0].copy()
    env_manager.global_maps['action_map'] = new_timestep.state.world.action_map.map[0].copy()
    env_manager.global_maps['dumpability_mask'] = new_timestep.state.world.dumpability_mask.map[0].copy()
    env_manager.global_maps['dumpability_mask_init'] = new_timestep.state.world.dumpability_mask_init.map[0].copy()
    env_manager.global_maps['padding_mask'] = new_timestep.state.world.padding_mask.map[0].copy()
    env_manager.global_maps['traversability_mask'] = new_timestep.state.world.traversability_mask.map[0].copy()
    env_manager.global_maps['trench_axes'] = new_timestep.state.world.trench_axes.copy()
    env_manager.global_maps['trench_type'] = new_timestep.state.world.trench_type.copy()
    
    # Store the new global timestep
    env_manager.global_timestep = new_timestep
    
    # Clear any existing partitions data to ensure fresh start
    env_manager.partitions = []
    env_manager.overlap_map = {}
    env_manager.overlap_regions = {}
    
    print(f"Environment reset to map {map_index}")
    print(f"New target map has {jnp.sum(env_manager.global_maps['target_map'] < 0)} dig targets")
    
def initialize_partitions_for_current_map(env_manager, config, model_params):
    """Initialize all partitions for the current map"""
    partition_states = {}
    partition_models = {}
    active_partitions = []

    num_partitions = len(env_manager.partitions)
    print(f"Number of partitions: {num_partitions}")

    # Initialize all partitions
    for partition_idx in range(num_partitions):
        try:
            print(f"Initializing partition {partition_idx}...")
                
            small_env_timestep = env_manager.initialize_small_environment(partition_idx)
                
            partition_states[partition_idx] = {
                'timestep': small_env_timestep,
                'prev_actions_rl': jnp.zeros((1, config.num_prev_actions), dtype=jnp.int32),
                'step_count': 0,
                'status': 'active',
                'rewards': [],
                'actions': [],
                'total_reward': 0.0,
            }
            active_partitions.append(partition_idx)

            partition_models[partition_idx] = {
                'model': load_neural_network(config, env_manager.small_env),
                'params': model_params.copy(),
                'prev_actions': jnp.zeros((1, config.num_prev_actions), dtype=jnp.int32)
            }
                        
        except Exception as e:
            print(f"Failed to initialize partition {partition_idx}: {e}")
            if partition_idx < len(env_manager.partitions):
                env_manager.partitions[partition_idx]['status'] = 'failed'

    if not active_partitions:
        print("No partitions could be initialized!")
        return None, None, None

    print(f"Successfully initialized {len(active_partitions)} partitions: {active_partitions}")
    return partition_states, partition_models, active_partitions

def init_llms(llm_model_key, llm_model_name, config, action_size, 
                     APP_NAME, USER_ID, SESSION_ID, MAP_SIZE, MAX_NUM_PARTITIONS):
    """
    Initialize LLMs using file-based prompts.
    """
    # Initialize prompt manager
    prompts = PromptManager(prompts_dir="llm/prompts")
    
    session_manager = SessionManager()
    
    if llm_model_key == "gpt":
        llm_model_name_extended = "openai/{}".format(llm_model_name)
    elif llm_model_key == "claude":
        llm_model_name_extended = "anthropic/{}".format(llm_model_name)
    else:
        llm_model_name_extended = llm_model_name
    
    print("Using model: ", llm_model_name_extended)

    # Load system messages from files
    system_message_master = prompts.get("partitioning_new", map_size=MAP_SIZE, max_partitions=MAX_NUM_PARTITIONS)
    
    system_message_delegation = prompts.get("delegation", observation="See current state")
    
    system_message_excavator = prompts.get("excavator_llm_simple")

    # Create agents
    if llm_model_key == "gemini":
        llm_partitioning_agent = Agent(
            name="PartitioningAgent",
            model=llm_model_name_extended,
            description="Master excavation coordinator for partitioning",
            instruction=system_message_master,
        )
        
        llm_delegation_agent = Agent(
            name="DelegationAgent", 
            model=llm_model_name_extended,
            description="Task delegation coordinator",
            instruction=system_message_delegation,
        )

        llm_excavator_agent = Agent(
            name="ExcavatorAgent",
            model=llm_model_name_extended,
            description="Excavator control agent",
            instruction=system_message_excavator,
        )
    else:
        llm_partitioning_agent = Agent(
            name="PartitioningAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description="Master excavation coordinator for partitioning",
            instruction=system_message_master,
        )
        
        llm_delegation_agent = Agent(
            name="DelegationAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description="Task delegation coordinator",
            instruction=system_message_delegation,
        )

        llm_excavator_agent = Agent(
            name="ExcavatorAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description="Excavator control agent",
            instruction=system_message_excavator,
        )


    # Partitioning session
    app_name_partitioning = f"{APP_NAME}_partitioning"
    user_id_partitioning = f"{USER_ID}_partitioning"
    session_id_partitioning = f"{SESSION_ID}_partitioning"
    
    session_service_partitioning = session_manager.create_agent_session(
        "PartitioningAgent",
        app_name_partitioning,
        user_id_partitioning,
        session_id_partitioning
    )

    # Delegation session
    app_name_delegation = f"{APP_NAME}_delegation"
    user_id_delegation = f"{USER_ID}_delegation"
    session_id_delegation = f"{SESSION_ID}_delegation"
    
    session_service_delegation = session_manager.create_agent_session(
        "DelegationAgent",
        app_name_delegation,
        user_id_delegation,
        session_id_delegation
    )

    # Excavator session
    app_name_excavator = f"{APP_NAME}_excavator"
    user_id_excavator = f"{USER_ID}_excavator"
    session_id_excavator = f"{SESSION_ID}_excavator"
    
    session_service_excavator = session_manager.create_agent_session(
        "ExcavatorAgent",
        app_name_excavator,
        user_id_excavator,
        session_id_excavator
    )

    #print("All sessions created successfully.")

    # CREATE RUNNERS WITH SESSION MANAGER
    runner_partitioning = session_manager.create_runner(
        llm_partitioning_agent,
        "PartitioningAgent",
        app_name_partitioning
    )
    
    runner_delegation = session_manager.create_runner(
        llm_delegation_agent,
        "DelegationAgent", 
        app_name_delegation
    )
    
    runner_excavator = session_manager.create_runner(
        llm_excavator_agent,
        "ExcavatorAgent",
        app_name_excavator
    )

    #print("All runners created successfully.")

    # Create LLM query object    
    llm_query = LLM_query(
        model_name=llm_model_name_extended,
        model=llm_model_key,
        system_message=system_message_excavator,
        action_size=action_size,
        session_id=session_id_excavator,
        runner=runner_excavator,
        user_id=user_id_excavator,
    )

    # Initialize previous actions
    prev_actions = None
    if config:
        import jax.numpy as jnp
        prev_actions = jnp.zeros(
            (1, config.num_prev_actions),
            dtype=jnp.int32
        )
    else:
        print("Warning: rl_config is None, prev_actions will not be initialized.")
    
    # Debug: List all sessions
    #session_manager.list_sessions()

    return (prompts, llm_query, runner_partitioning, runner_delegation, prev_actions, session_manager)

def get_delegation_prompt(prompts, current_observation, context="", ENABLE_INTERVENTION=False):
    """Get delegation prompt with current state."""
    try:
        obs_str = json.dumps({k: v.tolist() if hasattr(v, 'tolist') else str(v) 
                            for k, v in current_observation.items()}) if isinstance(current_observation, dict) else str(current_observation)
        if ENABLE_INTERVENTION:
            prompt = prompts.get("delegation", observation=obs_str)
        else:
            prompt = prompts.get("delegation_no_intervention", observation=obs_str)
        if context:
            prompt += f"\n\nAdditional context: {context}"
        return prompt
    except Exception as e:
        print(f"Error generating delegation prompt: {e}")
        return "Decide between 'delegate_to_rl' or 'delegate_to_llm'."

def get_excavator_prompt(prompts, direction, bucket_status, position):
    """Get excavator action prompt with current state."""
    try:
        return prompts.get("excavator_action", 
                          direction=direction, 
                          bucket_status=bucket_status, 
                          position=position)
    except Exception as e:
        print(f"Error generating excavator prompt: {e}")
        return "Choose the best action (0-6) for the current game state."
    
async def call_agent_async_master(query: str, image, runner, user_id, session_id, session_manager=None):
    """
    Fixed version of call_agent_async_master with better error handling and session verification.
    """
    print(f"\n>>> Calling agent with user_id: {user_id}, session_id: {session_id}")
    
    # Verify session exists if session_manager is provided
    if session_manager:
        session_info = session_manager.get_session_info(user_id, session_id)
        if not session_info:
            print(f"WARNING: Session {user_id}_{session_id} not found in session manager")
            # Try to recreate session if possible
            # This would require more context about the agent and app_name
    
    # Prepare the user's message in ADK format
    text = types.Part.from_text(text=query)
    parts = [text]
    
    if image is not None:
        image_data = encode_image(image)
        content_image = types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
        parts.append(content_image)

    user_content = types.Content(role='user', parts=parts)
    
    final_response_text = "Agent did not produce a final response."  # Default

    try:
        # Execute the agent with proper error handling
        async for event in runner.run_async(
            user_id=user_id, 
            session_id=session_id, 
            new_message=user_content
        ):
            print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}")
            
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
                
    except Exception as e:
        print(f"Error during agent execution: {e}")
        final_response_text = f"Error: {str(e)}"
    
    print(f"<<< Agent Response: {final_response_text}")
    return final_response_text

def setup_partitions_and_llm(map_index, ORIGINAL_MAP_SIZE, env_manager, config, llm_model_name, llm_model_key,
                                  APP_NAME, USER_ID, SESSION_ID, screen, USE_MANUAL_PARTITIONING=False,
                                  USE_IMAGE_PROMPT=False,MAX_NUM_PARTITIONS=4,USE_EXACT_NUMBER_OF_PARTITIONS=False, USE_RANDOM_PARTITIONING=False, sub_task_seed=58):
    """
    Setup_partitions_and_llm with proper session management.
    """

    action_size = 7
    target_map = env_manager.global_maps['target_map']
    
    if USE_MANUAL_PARTITIONING:
        sub_tasks_manual = compute_manual_subtasks(ORIGINAL_MAP_SIZE, MAX_NUM_PARTITIONS)
    elif USE_RANDOM_PARTITIONING:
        sub_tasks_manual = compute_random_subtasks_validated(
                ORIGINAL_MAP_SIZE, MAX_NUM_PARTITIONS, target_map,
                seed=sub_task_seed * (map_index+1), min_targets=1
            )
    else:
        sub_tasks_manual = compute_manual_subtasks(ORIGINAL_MAP_SIZE, MAX_NUM_PARTITIONS)

    # Initialize LLM agent with fixed session management
    (prompts, llm_query, runner_partitioning, runner_delegation, prev_actions, session_manager) = init_llms(
        llm_model_key, llm_model_name, config, action_size, 
        APP_NAME, USER_ID, f"{SESSION_ID}_map_{map_index}", ORIGINAL_MAP_SIZE, MAX_NUM_PARTITIONS
    )

    sub_tasks_llm = []
    
    # ALWAYS initialize partitions - either manual or LLM-generated
    if USE_MANUAL_PARTITIONING or USE_RANDOM_PARTITIONING:
        print("Using manually or random defined sub-tasks.")
        env_manager.initialize_with_fixed_overlaps(sub_tasks_manual)
    else:
        print("Calling LLM agent for partitioning decision...")

        game_state_image = capture_screen(screen)
        current_observation = env_manager.global_env.timestep.observation
        
        try:
            obs_dict = {k: v.tolist() for k, v in current_observation.items()}
            observation_str = json.dumps(obs_dict)
        except AttributeError:
            observation_str = str(current_observation)
        # Use file-based prompt
        if USE_IMAGE_PROMPT:
            if USE_EXACT_NUMBER_OF_PARTITIONS:
                prompt = prompts.get("partitioning_exact", 
                            map_size=ORIGINAL_MAP_SIZE, 
                            max_partitions=MAX_NUM_PARTITIONS) + "\n\nCurrent observation: See image"
            else:
                prompt = prompts.get("partitioning", 
                            map_size=ORIGINAL_MAP_SIZE, 
                            max_partitions=MAX_NUM_PARTITIONS) + "\n\nCurrent observation: See image"

        else:
            try:
                obs_dict = {k: v.tolist() for k, v in current_observation.items()}
                observation_str = json.dumps(obs_dict)
            except AttributeError:
                observation_str = str(current_observation)
            
            if USE_EXACT_NUMBER_OF_PARTITIONS:
                prompt = prompts.get("partitioning_exact", 
                            map_size=ORIGINAL_MAP_SIZE, 
                            max_partitions=MAX_NUM_PARTITIONS) + "\n\nCurrent observation: See image"
            else:
                prompt = prompts.get("partitioning", 
                            map_size=ORIGINAL_MAP_SIZE, 
                            max_partitions=MAX_NUM_PARTITIONS) + "\n\nCurrent observation: See image"

        try:
            user_id_partitioning = f"{USER_ID}_partitioning"
            session_id_partitioning = f"{SESSION_ID}_map_{map_index}_partitioning"
            
            if USE_IMAGE_PROMPT:
                response = asyncio.run(call_agent_async_master(
                    prompt, game_state_image, runner_partitioning, 
                    user_id_partitioning, session_id_partitioning, session_manager
                ))
            else:
                response = asyncio.run(call_agent_async_master(
                    prompt, None, runner_partitioning, 
                    user_id_partitioning, session_id_partitioning, session_manager
                ))
        
            llm_response_text = response

            try:
                sub_tasks_llm = extract_python_format_data(llm_response_text)
                print("Successfully parsed LLM response with tuples preserved")
            except ValueError as e:
                print(f"Extraction failed: {e}")
                sub_tasks_llm = sub_tasks_manual

        except Exception as adk_err:
            print(f"Error during PARTITIONING ADK agent communication: {adk_err}")
            sub_tasks_llm = sub_tasks_manual

        partition_validation = is_valid_region_list(sub_tasks_llm)
        
        if partition_validation:
            print("Using LLM-generated sub-tasks.")
            env_manager.initialize_with_fixed_overlaps(sub_tasks_llm)
        else:
            print("LLM-generated partitions invalid, falling back to manually defined sub-tasks.")
            env_manager.initialize_with_fixed_overlaps(sub_tasks_manual)

    return llm_query, runner_delegation, session_manager, prompts

def extract_subsurface(screen, x_start, y_start, width, height, ORIGINAL_MAP_SIZE, global_env_config, partition_idx):
    """Extract a subsurface from the screen."""
                           
    try:
        screen_width, screen_height = screen.get_size()
                                
        # Get the actual tile size from the environment
        # This should be available from your global_env_config
        env_tile_size = global_env_config.tile_size[0].item()  # From your existing code
                                
        # Calculate the rendering scale factor
        # This depends on how the environment renders to the screen
        render_scale = screen_width / (ORIGINAL_MAP_SIZE * env_tile_size)
                                
        # Convert game world coordinates to screen pixel coordinates
        screen_x_start = int(x_start * env_tile_size * render_scale)
        screen_y_start = int(y_start * env_tile_size * render_scale)
        screen_width_partition = int(width * env_tile_size * render_scale)
        screen_height_partition = int(height * env_tile_size * render_scale)
                                                            
        # Rest of the clamping and subsurface creation code remains the same...
        screen_x_start = max(0, min(screen_x_start, screen_width - 1))
        screen_y_start = max(0, min(screen_y_start, screen_height - 1))
        screen_width_partition = min(screen_width_partition, screen_width - screen_x_start)
        screen_height_partition = min(screen_height_partition, screen_height - screen_y_start)
                                
        if screen_width_partition <= 0 or screen_height_partition <= 0:
            print(f"    Warning: Invalid partition size, using fallback")
            subsurface = screen.subsurface((0, 0, min(64, screen_width), min(64, screen_height)))
        else:
            subsurface = screen.subsurface((screen_x_start, screen_y_start, screen_width_partition, screen_height_partition))
                                    
    except ValueError as e:
        print(f"Error extracting subsurface for partition {partition_idx}: {e}")
        fallback_size = min(64, screen_width, screen_height)
        subsurface = screen.subsurface((0, 0, fallback_size, fallback_size))

    return subsurface

def capture_screen(surface):
    """Captures the current screen and converts it to an image format."""
    img_array = pg.surfarray.array3d(surface)
    #img_array = np.rot90(img_array, k=3)  # Rotate if needed
    img_array = np.transpose(img_array, (1, 0, 2))  # Correct rotation

    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array

def save_video(frames, output_path, fps=1):
    """Saves a list of frames as a video."""
    if len(frames) == 0:
        print("No frames to save.")
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

def extract_bucket_status(state):
    """
    Extract the bucket status from the state.

    Args:
        state: The current State object.

    Returns:
        str: The bucket status ('loaded' or 'empty').
    """
    # Access the bucket status from the agent's state
    bucket_status = state.agent.agent_state.loaded

    # Map the status to a human-readable string
    return "loaded" if bucket_status else "empty"

def base_orientation_to_direction(angle_base):
    """
    Convert the base orientation value (0-3) to a cardinal direction.

    Args:
        angle_base (int or JAX array): The base orientation value.

    Returns:
        str: The corresponding cardinal direction ('up', 'right', 'down', 'left').
    """
    # Convert JAX array to a Python scalar if necessary
    if isinstance(angle_base, jax.Array):
        angle_base = angle_base.item()

    # Map orientation to cardinal direction
    direction_map = {
        0: "right",
        3: "up",
        7: "left",
        11: "down"
    }
    return direction_map.get(angle_base, "unknown")  # Default to 'unknown' if invalid

def extract_base_orientation(state):
    """
    Extract the excavator's base orientation from the state and convert it to a cardinal direction.

    Args:
        state: The current State object.

    Returns:
        A dictionary containing the base angle and its corresponding cardinal direction.
    """
    # Extract the base angle
    angle_base = state.agent.agent_state.angle_base

    # Convert the base angle to a cardinal direction
    direction = base_orientation_to_direction(angle_base)

    return {
        "angle_base": angle_base,
        "direction": direction,
    }    

def load_experiment_constants(config_file="llm/config_llm.yaml"):
    """
    Load all experiment constants from YAML file.
    Returns a namespace object with all your constants as attributes.
    
    Usage:
        Replace this:
            FORCE_DELEGATE_TO_RL = True
            FORCE_DELEGATE_TO_LLM = False
            ...
        
        With this:
            constants = load_experiment_constants()
            FORCE_DELEGATE_TO_RL = constants.force_delegate_to_rl
            FORCE_DELEGATE_TO_LLM = constants.force_delegate_to_llm
            ...
    """
    
    # Default values (same as your original constants)
    defaults = {
        'force_delegate_to_rl': True,
        'force_delegate_to_llm': False,
        'llm_call_frequency': 15,
        'use_manual_partitioning': True,
        'max_num_partitions': 2,
        'visualize_partitions': True,
        'use_image_prompt': True,
        'app_name': "ExcavatorGameApp",
        'user_id': "user_1",
        'session_id': "session_001",
        'grid_rendering': True,
        'original_map_size': 128,
        'use_rendering': True,
        'use_display': True,
        'enable_intervention': True,
        'intervention_check_frequency': 15,
        'stuck_detection_window': 10,
        'min_reward_threshold': 0.001,
        'use_random_partitioning': False,
        'use_exact_number_of_partitions': False,
        'save_video': False,
        'fps': 30,
        'compute_bench_stats': False,
        'use_exclusive_assignment': False,  # New constant for exclusive assignment
    }
    
    # Try to load from YAML file
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                # Update defaults with loaded values
                defaults.update(config)
                #print(f"✅ Configuration loaded from {config_file}")
        except Exception as e:
            print(f"⚠️  Error loading {config_file}: {e}. Using defaults.")
    else:
        print(f"⚠️  Config file {config_file} not found. Using defaults.")
    
    # Convert to namespace for easy attribute access
    class ConfigNamespace:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
        
        def __repr__(self):
            attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
            return f"ConfigNamespace({', '.join(attrs)})"
    
    return ConfigNamespace(defaults)

def setup_experiment_config(config_file="llm/config_llm.yaml"):
    """
    Direct replacement for your constant definitions.
    
    Replace this block in your code:
        FORCE_DELEGATE_TO_RL = True
        FORCE_DELEGATE_TO_LLM = False
        LLM_CALL_FREQUENCY = 15
        # ... etc
    
    With this single line:
        FORCE_DELEGATE_TO_RL, FORCE_DELEGATE_TO_LLM, LLM_CALL_FREQUENCY, ... = setup_experiment_config()
    """
    constants = load_experiment_constants(config_file)
    
    return (
        constants.force_delegate_to_rl,     # FORCE_DELEGATE_TO_RL
        constants.force_delegate_to_llm,    # FORCE_DELEGATE_TO_LLM  
        constants.llm_call_frequency,       # LLM_CALL_FREQUENCY
        constants.use_manual_partitioning,  # USE_MANUAL_PARTITIONING
        constants.max_num_partitions,       # MAX_NUM_PARTITIONS
        constants.visualize_partitions,     # VISUALIZE_PARTITIONS
        constants.use_image_prompt,         # USE_IMAGE_PROMPT
        constants.app_name,                 # APP_NAME
        constants.user_id,                  # USER_ID
        constants.session_id,               # SESSION_ID
        constants.grid_rendering,           # GRID_RENDERING
        constants.original_map_size,        # ORIGINAL_MAP_SIZE
        constants.use_rendering,            # USE_RENDERING
        constants.use_display,               # USE_DISPLAY
        constants.enable_intervention,
        constants.intervention_check_frequency,
        constants.stuck_detection_window,
        constants.min_reward_threshold,
        constants.use_random_partitioning,  # USE_RANDOM_PARTITIONING
        constants.use_exact_number_of_partitions,
        constants.save_video,           # SAVE_VIDEO
        constants.fps,
        constants.compute_bench_stats,
        constants.use_exclusive_assignment,  # USE_EXCLUSIVE_ASSIGNMENT
    )

def detect_stuck_excavator(partition_state, threshold_steps=10, min_reward_threshold=0.001):
    """
    Detect if an excavator is stuck based on recent performance.
    
    Args:
        partition_state: Current partition state dictionary
        threshold_steps: Number of recent steps to analyze
        min_reward_threshold: Minimum reward expected in the period
    
    Returns:
        dict: Stuck detection result with details
    """
    step_count = partition_state['step_count']
    rewards = partition_state['rewards']
    actions = partition_state['actions']
    
    # Need minimum steps to evaluate
    if step_count < threshold_steps:
        return {
            'is_stuck': False,
            'reason': 'insufficient_data',
            'details': f'Only {step_count} steps, need {threshold_steps}'
        }
    
    # Analyze recent performance
    recent_rewards = rewards[-threshold_steps:] if len(rewards) >= threshold_steps else rewards
    recent_actions = actions[-threshold_steps:] if len(actions) >= threshold_steps else actions
    
    # Check 1: Very low or negative rewards
    total_recent_reward = sum(recent_rewards)
    if total_recent_reward < min_reward_threshold:
        return {
            'is_stuck': True,
            'reason': 'low_reward',
            'details': f'Total reward in last {len(recent_rewards)} steps: {total_recent_reward:.4f}'
        }
    
    # Check 2: Repetitive actions (agent going in circles)
    if len(recent_actions) >= 6:
        action_sequence = [int(a[0]) if hasattr(a, '__getitem__') else int(a) for a in recent_actions[-6:]]
        # Check for simple repetitive patterns
        if len(set(action_sequence)) <= 2:  # Only using 2 or fewer different actions
            return {
                'is_stuck': True,
                'reason': 'repetitive_actions',
                'details': f'Recent actions: {action_sequence}'
            }
    
    # Check 3: No progress (consistently getting 0 rewards)
    zero_reward_count = sum(1 for r in recent_rewards if abs(r) < 0.0001)
    if zero_reward_count >= threshold_steps * 0.8:  # 80% of recent steps had no reward
        return {
            'is_stuck': True,
            'reason': 'no_progress',
            'details': f'{zero_reward_count}/{len(recent_rewards)} recent steps had zero reward'
        }
    
    return {
        'is_stuck': False,
        'reason': 'performing_well',
        'details': f'Recent reward: {total_recent_reward:.4f}, unique actions: {len(set([int(a[0]) if hasattr(a, "__getitem__") else int(a) for a in recent_actions]))}'
    }

def get_intervention_action(partition_state, stuck_info, action_type):
    """
    Get an intervention action to help unstuck the agent.
    
    Args:
        partition_state: Current partition state
        stuck_info: Result from detect_stuck_excavator()
        action_type: Action type class (TrackedAction or WheeledAction)
    
    Returns:
        jnp.array: Action to take
    """
    # Import action types
    from terra.actions import TrackedActionType, WheeledActionType
    
    # Determine available actions based on action type
    if action_type.__name__ == 'TrackedAction':
        FORWARD = TrackedActionType.FORWARD
        BACKWARD = TrackedActionType.BACKWARD
        DO = TrackedActionType.DO
        turn_actions = []  # Tracked vehicles don't have turn actions
    else:  # WheeledAction
        FORWARD = WheeledActionType.FORWARD
        BACKWARD = WheeledActionType.BACKWARD
        CLOCK = WheeledActionType.CLOCK
        ANTICLOCK = WheeledActionType.ANTICLOCK
        DO = WheeledActionType.DO
        turn_actions = [CLOCK, ANTICLOCK]
    
    # Get recent actions to avoid repeating them
    recent_actions = partition_state['actions'][-5:] if len(partition_state['actions']) >= 5 else partition_state['actions']
    recent_action_values = [int(a[0]) if hasattr(a, '__getitem__') else int(a) for a in recent_actions]
    
    intervention_action = None
    
    if stuck_info['reason'] == 'repetitive_actions':
        # Agent is repeating actions - try something different
        print(f"    🔧 INTERVENTION: Breaking repetitive pattern")
        
        # If mostly moving forward/backward, try turning (for wheeled) or DO action
        if FORWARD in recent_action_values or BACKWARD in recent_action_values:
            if turn_actions and CLOCK not in recent_action_values:
                intervention_action = CLOCK
            elif turn_actions and ANTICLOCK not in recent_action_values:
                intervention_action = ANTICLOCK
            else:
                intervention_action = DO  # Try digging/dumping
        else:
            # Try moving if not moving recently
            intervention_action = FORWARD
    
    elif stuck_info['reason'] == 'low_reward' or stuck_info['reason'] == 'no_progress':
        # Agent making no progress - try strategic actions
        print(f"    🔧 INTERVENTION: Addressing low progress")
        
        # Cycle through: turn -> move -> dig -> turn
        step_in_cycle = partition_state['step_count'] % 4
        
        if step_in_cycle == 0 and turn_actions:
            intervention_action = CLOCK
        elif step_in_cycle == 1:
            intervention_action = FORWARD
        elif step_in_cycle == 2:
            intervention_action = DO
        else:
            if turn_actions:
                intervention_action = ANTICLOCK
            else:
                intervention_action = BACKWARD
    
    else:
        # Default intervention - try a different direction
        print(f"    🔧 INTERVENTION: General unstuck attempt")
        if turn_actions:
            intervention_action = CLOCK  # Turn to face new direction
        else:
            intervention_action = BACKWARD  # Back up
    
    # Convert to proper format
    intervention_action_val = int(intervention_action)
    print(f"    🔧 INTERVENTION ACTION: {intervention_action_val} (reason: {stuck_info['reason']})")
    
    return jnp.array([intervention_action_val], dtype=jnp.int32)

def should_intervene(partition_state, active_partitions, intervention_frequency=15):
    """
    Decide if intervention should be considered for this partition.
    
    Args:
        partition_state: Current partition state
        active_partitions: List of active partitions
        intervention_frequency: How often to check for intervention
    
    Returns:
        bool: True if intervention should be considered
    """
    # Only check every N steps to avoid over-intervening
    if partition_state['step_count'] % intervention_frequency != 0:
        return False
    
    # Don't intervene too early
    if partition_state['step_count'] < 10:
        return False
    
    # Check if stuck
    stuck_info = detect_stuck_excavator(partition_state)
    return stuck_info['is_stuck']

def check_partition_has_targets(target_map, region_coords, min_targets=1):
    """
    Check if a partition region contains dig targets.
    
    Args:
        target_map: Global target map (jnp.ndarray)
        region_coords: (y_start, x_start, y_end, x_end) tuple
        min_targets: Minimum number of dig targets required
    
    Returns:
        dict: Information about targets in the partition
    """
    y_start, x_start, y_end, x_end = region_coords
    
    # Extract the region from the target map
    region_slice = (slice(y_start, y_end + 1), slice(x_start, x_end + 1))
    region_data = target_map[region_slice]
    
    # Count dig targets (-1) and dump targets (1)
    dig_targets = jnp.sum(region_data == -1)
    dump_targets = jnp.sum(region_data == 1)
    free_space = jnp.sum(region_data == 0)
    
    has_enough_targets = dig_targets >= min_targets
    
    return {
        'has_targets': has_enough_targets,
        'dig_targets': int(dig_targets),
        'dump_targets': int(dump_targets),
        'free_space': int(free_space),
        'total_cells': region_data.size,
        'region_coords': region_coords
    }

def filter_empty_partitions(partitions, target_map, min_targets=1):
    """
    Filter out partitions that don't have enough dig targets.
    
    Args:
        partitions: List of partition dictionaries
        target_map: Global target map from env_manager
        min_targets: Minimum dig targets required per partition
    
    Returns:
        tuple: (filtered_partitions, partition_stats)
    """
    valid_partitions = []
    partition_stats = []
    
    for partition in partitions:
        region_coords = partition['region_coords']
        stats = check_partition_has_targets(target_map, region_coords, min_targets)
        partition_stats.append(stats)
        
        if stats['has_targets']:
            valid_partitions.append(partition)
            print(f"  Partition {partition['id']}: {stats['dig_targets']} dig targets, {stats['dump_targets']} dump targets")
        else:
            print(f"  Partition {partition['id']}: Only {stats['dig_targets']} dig targets (minimum: {min_targets}) - SKIPPED")
    
    return valid_partitions, partition_stats

def compute_random_subtasks_validated(ORIGINAL_MAP_SIZE, NUM_PARTITIONS, target_map, 
                                     seed=None, min_targets=1, max_attempts=100):
    """
    Compute random subtasks and validate they contain dig targets.
    Will retry with different random splits if partitions are empty.
    
    Args:
        ORIGINAL_MAP_SIZE: Size of the original map (64 or 128)
        NUM_PARTITIONS: Number of partitions (1, 2, or 4)
        target_map: Global target map from env_manager
        seed: Random seed for reproducibility
        min_targets: Minimum dig targets required per partition
        max_attempts: Maximum attempts to find valid partitions
    
    Returns:
        List of valid partition dictionaries
    """
    if seed is not None:
        random.seed(seed)
    
    if ORIGINAL_MAP_SIZE not in [64, 128]:
        raise ValueError(f"Unsupported ORIGINAL_MAP_SIZE: {ORIGINAL_MAP_SIZE}. Must be 64 or 128.")
    
    if NUM_PARTITIONS not in [1, 2, 4]:
        raise ValueError("Invalid number of partitions. Must be 1, 2 or 4.")
    
    min_targets = int(jnp.sum(target_map == -1) * 0.3)
    print(f"Minimum targets per partition: {min_targets}")
    
    print(f"\nGenerating {NUM_PARTITIONS} random partitions with target validation...")
    
    # For single partition, just return the full map if it has targets
    if NUM_PARTITIONS == 1:
        full_partition = [{
            'id': 0, 
            'region_coords': (0, 0, ORIGINAL_MAP_SIZE - 1, ORIGINAL_MAP_SIZE - 1), 
            'start_pos': (ORIGINAL_MAP_SIZE // 2, ORIGINAL_MAP_SIZE // 2), 
            'start_angle': 0, 
            'status': 'pending'
        }]
        
        valid_partitions, _ = filter_empty_partitions(full_partition, target_map, min_targets)
        if valid_partitions:
            return valid_partitions
        else:
            raise ValueError("The entire map has no dig targets!")
    
    best_partitions = []
    best_valid_count = 0
    
    for attempt in range(max_attempts):
        print(f"  Attempt {attempt + 1}/{max_attempts}...")
        
        # Generate random partitions
        if NUM_PARTITIONS == 2:
            partitions = _generate_random_2_partitions(ORIGINAL_MAP_SIZE)
        elif NUM_PARTITIONS == 4:
            partitions = _generate_random_4_partitions(ORIGINAL_MAP_SIZE)
        
        # Validate partitions
        valid_partitions, stats = filter_empty_partitions(partitions, target_map, min_targets)
        
        print(f"    Generated {len(partitions)} partitions, {len(valid_partitions)} valid")
        
        # Keep track of the best result so far
        if len(valid_partitions) > best_valid_count:
            best_partitions = valid_partitions.copy()
            best_valid_count = len(valid_partitions)
        
        # If we got enough valid partitions, we're done
        #if len(valid_partitions) >= max(1, NUM_PARTITIONS // 2):  # At least half the requested partitions
        if len(valid_partitions) == NUM_PARTITIONS:
            print(f"  Found {len(valid_partitions)} valid partitions after {attempt + 1} attempts")
            return valid_partitions
    
    # If we couldn't find enough good partitions, return the best we found
    if best_partitions:
        print(f"  Returning best result: {len(best_partitions)} valid partitions (out of {NUM_PARTITIONS} requested)")
        return best_partitions
    else:
        # Fallback: return full map as single partition
        print(f"  No valid partitions found, falling back to single full-map partition")
        return [{
            'id': 0, 
            'region_coords': (0, 0, ORIGINAL_MAP_SIZE - 1, ORIGINAL_MAP_SIZE - 1), 
            'start_pos': (ORIGINAL_MAP_SIZE // 2, ORIGINAL_MAP_SIZE // 2), 
            'start_angle': 0, 
            'status': 'pending'
        }]

def _generate_random_2_partitions(ORIGINAL_MAP_SIZE):
    """Generate 2 random partitions."""
    is_vertical = random.choice([True, False])
    
    if is_vertical:
        # Vertical split
        min_split = int(ORIGINAL_MAP_SIZE * 0.3)
        max_split = int(ORIGINAL_MAP_SIZE * 0.7)
        split_x = random.randint(min_split, max_split)
        
        start_y = ORIGINAL_MAP_SIZE // 2
        start_x1 = split_x // 2
        start_x2 = split_x + (ORIGINAL_MAP_SIZE - split_x) // 2
        
        return [
            {
                'id': 0,
                'region_coords': (0, 0, ORIGINAL_MAP_SIZE - 1, split_x - 1),
                'start_pos': (start_y, start_x1),
                'start_angle': 0,
                'status': 'pending'
            },
            {
                'id': 1,
                'region_coords': (0, split_x, ORIGINAL_MAP_SIZE - 1, ORIGINAL_MAP_SIZE - 1),
                'start_pos': (start_y, start_x2),
                'start_angle': 0,
                'status': 'pending'
            }
        ]
    else:
        # Horizontal split
        min_split = int(ORIGINAL_MAP_SIZE * 0.3)
        max_split = int(ORIGINAL_MAP_SIZE * 0.7)
        split_y = random.randint(min_split, max_split)
        
        start_x = ORIGINAL_MAP_SIZE // 2
        start_y1 = split_y // 2
        start_y2 = split_y + (ORIGINAL_MAP_SIZE - split_y) // 2
        
        return [
            {
                'id': 0,
                'region_coords': (0, 0, split_y - 1, ORIGINAL_MAP_SIZE - 1),
                'start_pos': (start_y1, start_x),
                'start_angle': 0,
                'status': 'pending'
            },
            {
                'id': 1,
                'region_coords': (split_y, 0, ORIGINAL_MAP_SIZE - 1, ORIGINAL_MAP_SIZE - 1),
                'start_pos': (start_y2, start_x),
                'start_angle': 0,
                'status': 'pending'
            }
        ]

def _generate_random_4_partitions(ORIGINAL_MAP_SIZE):
    """Generate 4 random partitions in a 2x2 grid."""
    min_split = int(ORIGINAL_MAP_SIZE * 0.3)
    max_split = int(ORIGINAL_MAP_SIZE * 0.7)
    
    split_x = random.randint(min_split, max_split)
    split_y = random.randint(min_split, max_split)
    
    # Calculate start positions for each quadrant
    start_x1 = split_x // 2
    start_x2 = split_x + (ORIGINAL_MAP_SIZE - split_x) // 2
    start_y1 = split_y // 2
    start_y2 = split_y + (ORIGINAL_MAP_SIZE - split_y) // 2
    
    return [
        {
            'id': 0,
            'region_coords': (0, 0, split_y - 1, split_x - 1),
            'start_pos': (start_y1, start_x1),
            'start_angle': 0,
            'status': 'pending'
        },
        {
            'id': 1,
            'region_coords': (0, split_x, split_y - 1, ORIGINAL_MAP_SIZE - 1),
            'start_pos': (start_y1, start_x2),
            'start_angle': 0,
            'status': 'pending'
        },
        {
            'id': 2,
            'region_coords': (split_y, 0, ORIGINAL_MAP_SIZE - 1, split_x - 1),
            'start_pos': (start_y2, start_x1),
            'start_angle': 0,
            'status': 'pending'
        },
        {
            'id': 3,
            'region_coords': (split_y, split_x, ORIGINAL_MAP_SIZE - 1, ORIGINAL_MAP_SIZE - 1),
            'start_pos': (start_y2, start_x2),
            'start_angle': 0,
            'status': 'pending'
        }
    ]

def extract_positions(state):
    """
    Extract the current base position and target position from the game state.

    Args:
        state: The current game state object.

    Returns:
        A tuple containing:
        - current_position: A dictionary with the current base position (x, y).
        - target_position: A dictionary with the target position (x, y), or None if not available.
        
    """

    # Extract th11e current base position
    current_position = {
        "x": state.agent.agent_state.pos_base[0][0],
        "y": state.agent.agent_state.pos_base[0][1]
    }

    # Extract the target position from the target_map if available
    target_positions = []

    for x in range(state.world.target_map.map.shape[1]):  # Iterate over rows
        for y in range(state.world.target_map.map.shape[2]):  # Iterate over columns
            if state.world.target_map.map[0, x, y] == -1:  # Access the value at (0, x, y)
                target_positions.append((x, y))
    
    # # Convert positions to tuples
    start = (int(current_position["x"]), int(current_position["y"]))

    return start, target_positions

def find_nearest_target(start, target_positions):
    """
    Find the nearest target position to the starting point.

    Args:
        start (tuple): The starting position as (x, y).
        target_positions (list of tuples): A list of target positions as (x, y).

    Returns:
        tuple: The nearest target position as (x, y), or None if the list is empty.
    """
    if not target_positions:
        return None

    # Calculate the Euclidean distance to each target and find the nearest one
    nearest_target = min(target_positions, key=lambda target: (target[0] - start[0])**2 + (target[1] - start[1])**2)
    return nearest_target