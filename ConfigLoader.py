import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Test loading
    try:
        config = load_config()
        print("Config loaded successfully:")
        print(config)
    except Exception as e:
        print(f"Error loading config: {e}")
