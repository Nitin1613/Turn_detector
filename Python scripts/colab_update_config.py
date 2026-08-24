# Quick config update for Colab - Run this cell ONCE

import yaml

print("Updating config for Colab GPU...")

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update device to use GPU (Colab has GPU)
config['inference']['device'] = 'cuda'

# Save
with open('configs/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("✓ Config updated to use GPU")
print("\nYou can now:")
print("1. Run the split creation cell")
print("2. Then run: !python train.py")
