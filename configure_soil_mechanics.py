#!/usr/bin/env python3
"""
Quick configuration script to optimize Terra soil mechanics for different use cases.
Use this to easily switch between training speed and realism.
"""

import os
import sys
import argparse

def update_soil_mechanics_config(mode='fast'):
    """Update the soil mechanics configuration in state.py"""
    
    state_file_path = '/cluster/home/alesweber/TerraProject/terra/terra/state.py'
    
    # Configuration mappings
    configs = {
        'fast': {
            'USE_SIMPLIFIED_SOIL_MECHANICS': True,
            'ENABLE_SOIL_MECHANICS_IN_TRAINING': False,
            'description': 'Maximum speed - no soil mechanics during training'
        },
        'balanced': {
            'USE_SIMPLIFIED_SOIL_MECHANICS': True,
            'ENABLE_SOIL_MECHANICS_IN_TRAINING': True,
            'description': 'Good speed + some realism - simplified soil mechanics'
        },
        'realistic': {
            'USE_SIMPLIFIED_SOIL_MECHANICS': False,
            'ENABLE_SOIL_MECHANICS_IN_TRAINING': True,
            'description': 'Full realism - complete soil mechanics (slower)'
        }
    }
    
    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Options: {list(configs.keys())}")
    
    config = configs[mode]
    
    print(f"🔧 Configuring Terra for '{mode}' mode...")
    print(f"📄 {config['description']}")
    
    try:
        # Read the current file
        with open(state_file_path, 'r') as f:
            content = f.read()
        
        # Update the configuration flags
        lines = content.split('\n')
        updated_lines = []
        
        for line in lines:
            if 'USE_SIMPLIFIED_SOIL_MECHANICS = ' in line:
                updated_lines.append(f"USE_SIMPLIFIED_SOIL_MECHANICS = {config['USE_SIMPLIFIED_SOIL_MECHANICS']}  # Set by configure_soil_mechanics.py")
            elif 'ENABLE_SOIL_MECHANICS_IN_TRAINING = ' in line:
                updated_lines.append(f"ENABLE_SOIL_MECHANICS_IN_TRAINING = {config['ENABLE_SOIL_MECHANICS_IN_TRAINING']}  # Set by configure_soil_mechanics.py")
            else:
                updated_lines.append(line)
        
        # Write the updated content
        with open(state_file_path, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print(f"✅ Successfully configured for '{mode}' mode")
        print(f"📊 Settings:")
        print(f"   - Simplified soil mechanics: {config['USE_SIMPLIFIED_SOIL_MECHANICS']}")
        print(f"   - Training soil mechanics: {config['ENABLE_SOIL_MECHANICS_IN_TRAINING']}")
        
        # Print expected performance impact
        if mode == 'fast':
            print(f"🚀 Expected speedup: 5-10x faster compilation")
        elif mode == 'balanced':
            print(f"⚖️ Expected speedup: 2-5x faster compilation")
        else:
            print(f"🐌 Slower compilation but full soil realism")
            
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        return False
    
    return True

def print_current_config():
    """Print the current soil mechanics configuration"""
    
    state_file_path = '/cluster/home/alesweber/TerraProject/terra/terra/state.py'
    
    try:
        with open(state_file_path, 'r') as f:
            content = f.read()
        
        simplified = None
        training = None
        
        for line in content.split('\n'):
            if 'USE_SIMPLIFIED_SOIL_MECHANICS = ' in line:
                simplified = 'True' in line
            elif 'ENABLE_SOIL_MECHANICS_IN_TRAINING = ' in line:
                training = 'True' in line
        
        print("📊 Current Soil Mechanics Configuration:")
        print(f"   - Simplified soil mechanics: {simplified}")
        print(f"   - Training soil mechanics: {training}")
        
        # Determine mode
        if not training:
            mode = 'fast'
        elif simplified and training:
            mode = 'balanced'
        elif not simplified and training:
            mode = 'realistic'
        else:
            mode = 'unknown'
        
        print(f"   - Current mode: {mode}")
        
    except Exception as e:
        print(f"❌ Failed to read current configuration: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Configure Terra soil mechanics for different use cases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python configure_soil_mechanics.py --mode fast      # Maximum training speed
  python configure_soil_mechanics.py --mode balanced  # Good speed + realism  
  python configure_soil_mechanics.py --mode realistic # Full realism
  python configure_soil_mechanics.py --status         # Show current config
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['fast', 'balanced', 'realistic'],
        help='Soil mechanics mode to configure'
    )
    
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show current configuration without changing anything'
    )
    
    args = parser.parse_args()
    
    print("🌍 Terra Soil Mechanics Configurator")
    print("="*50)
    
    if args.status:
        print_current_config()
    elif args.mode:
        success = update_soil_mechanics_config(args.mode)
        if success:
            print(f"\n💡 Tip: Run training with these settings for optimal {args.mode} performance")
            if args.mode == 'fast':
                print("💨 You should see significant speedup in compilation time!")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 