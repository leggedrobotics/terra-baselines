import os
import glob
from PIL import Image
import re
from typing import List, Tuple, Optional

def create_gif_from_traversability_pngs(
    folder_path: str = "before_RL",
    output_gif_path: str = "traversability_animation.gif",
    duration: int = 500,
    loop: int = 0,
    partition_range: Tuple[int, int] = (0, 3)
) -> bool:
    """
    Creates a GIF from traversability partition PNG files arranged in a 2x2 grid layout.
    
    Args:
        folder_path: Path to folder containing PNG files (default: "before_RL")
        output_gif_path: Output path for the GIF file
        duration: Duration between frames in milliseconds (default: 500ms)
        loop: Number of loops (0 = infinite loop)
        partition_range: Tuple of (min_partition, max_partition) inclusive
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        # Ensure folder exists
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist")
            return False
        
        # Get all unique step numbers across all partitions
        pattern = os.path.join(folder_path, "traversability_partition_*.png")
        png_files = glob.glob(pattern)
        
        if not png_files:
            print(f"No PNG files found matching pattern in '{folder_path}'")
            return False
        
        # Organize files by step and partition
        step_partition_map = {}
        
        for file_path in png_files:
            filename = os.path.basename(file_path)
            match = re.match(r'traversability_partition_(\d+)_step_(\d+)\.png', filename)
            if match:
                partition_x = int(match.group(1))
                step_y = int(match.group(2))
                
                # Filter by partition range
                if partition_range[0] <= partition_x <= partition_range[1]:
                    if step_y not in step_partition_map:
                        step_partition_map[step_y] = {}
                    step_partition_map[step_y][partition_x] = file_path
        
        if not step_partition_map:
            print(f"No files found with partitions in range {partition_range}")
            return False
        
        # Get sorted list of steps
        steps = sorted(step_partition_map.keys())
        partitions = list(range(partition_range[0], partition_range[1] + 1))
        
        print(f"Found {len(steps)} steps and {len(partitions)} partitions")
        
        # Load a sample image to get dimensions
        sample_img = None
        for step in steps:
            for partition in partitions:
                if partition in step_partition_map[step]:
                    sample_img = Image.open(step_partition_map[step][partition])
                    break
            if sample_img:
                break
        
        if not sample_img:
            print("Could not load any sample image")
            return False
        
        img_width, img_height = sample_img.size
        
        # 2x2 grid layout for 4 partitions (0-3)
        grid_cols = 2
        grid_rows = 2
        grid_width = img_width * grid_cols
        grid_height = img_height * grid_rows
        
        # Grid position mapping: partition -> (row, col)
        # Partition 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
        grid_positions = {
            0: (0, 0),  # top-left
            1: (0, 1),  # top-right
            2: (1, 0),  # bottom-left
            3: (1, 1),  # bottom-right
        }
        
        # Create frames for each step
        grid_frames = []
        
        for step in steps:
            # Create blank grid image
            grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
            
            # Place each partition in its grid position
            for partition in partitions:
                if partition in step_partition_map[step]:
                    try:
                        # Load partition image
                        img = Image.open(step_partition_map[step][partition])
                        
                        # Convert to RGB if necessary
                        if img.mode == 'RGBA':
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[-1])
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Calculate position in grid
                        row, col = grid_positions[partition]
                        x_pos = col * img_width
                        y_pos = row * img_height
                        
                        # Paste into grid
                        grid_img.paste(img, (x_pos, y_pos))
                        
                    except Exception as e:
                        print(f"Error loading partition {partition}, step {step}: {e}")
                        continue
            
            grid_frames.append(grid_img)
            print(f"Created grid frame for step {step}")
        
        if not grid_frames:
            print("No grid frames could be created")
            return False
        
        # Create GIF
        print(f"Creating GIF with {len(grid_frames)} frames in 2x2 grid layout...")
        grid_frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=grid_frames[1:],
            duration=duration,
            loop=loop,
            optimize=True
        )
        
        print(f"Grid GIF created successfully: {output_gif_path}")
        print(f"Grid layout: Partition 0 (top-left), 1 (top-right), 2 (bottom-left), 3 (bottom-right)")
        return True
        
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return False


def create_separate_gifs_by_partition(
    folder_path: str = "before_RL",
    output_dir: str = "gifs",
    duration: int = 500,
    loop: int = 0,
            partition_range: Tuple[int, int] = (0, 3)
) -> bool:
    """
    Creates separate GIF files for each partition.
    
    Args:
        folder_path: Path to folder containing PNG files
        output_dir: Directory to save GIF files
        duration: Duration between frames in milliseconds
        loop: Number of loops (0 = infinite loop)
        partition_range: Tuple of (min_partition, max_partition) inclusive
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each partition separately
        success_count = 0
        
        for partition_x in range(partition_range[0], partition_range[1] + 1):
            # Find files for this partition
            pattern = os.path.join(folder_path, f"traversability_partition_{partition_x}_step_*.png")
            partition_files = glob.glob(pattern)
            
            if not partition_files:
                print(f"No files found for partition {partition_x}")
                continue
            
            # Sort by step number
            file_info = []
            for file_path in partition_files:
                filename = os.path.basename(file_path)
                match = re.match(rf'traversability_partition_{partition_x}_step_(\d+)\.png', filename)
                if match:
                    step_y = int(match.group(1))
                    file_info.append((step_y, file_path))
            
            file_info.sort(key=lambda x: x[0])  # Sort by step
            
            # Load images for this partition
            images = []
            for step_y, file_path in file_info:
                try:
                    img = Image.open(file_path)
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    images.append(img)
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
            
            if images:
                # Create GIF for this partition
                output_path = os.path.join(output_dir, f"traversability_partition_{partition_x}.gif")
                images[0].save(
                    output_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration,
                    loop=loop,
                    optimize=True
                )
                
                print(f"Created GIF for partition {partition_x}: {output_path} ({len(images)} frames)")
                success_count += 1
        
        print(f"Successfully created {success_count} GIF files")
        return success_count > 0
        
    except Exception as e:
        print(f"Error creating separate GIFs: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Option 1: Create a single GIF with all partitions and steps
    print("Creating single GIF with all partitions...")
    create_gif_from_traversability_pngs(
        folder_path="before_RL",
        output_gif_path="all_traversability_partitions.gif",
        duration=300,  # 300ms between frames
        partition_range=(0, 4)
    )
    
    print("\n" + "="*50 + "\n")
    
    # Option 2: Create separate GIFs for each partition
    print("Creating separate GIFs for each partition...")
    create_separate_gifs_by_partition(
        folder_path="before_RL",
        output_dir="partition_gifs",
        duration=500,  # 500ms between frames
        partition_range=(0, 3)
    )