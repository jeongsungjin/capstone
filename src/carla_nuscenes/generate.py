from carla_nuscenes.generator import Generator
import os
import yaml

config_path = "./configs/config.yaml"

# Optional dependency: pyyaml-include. Fallback to simple !include handler if missing.
try:
    from yamlinclude import YamlIncludeConstructor  # type: ignore
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
except Exception:
    base_dir = os.path.dirname(os.path.abspath(config_path))

    def _include_constructor(loader: yaml.Loader, node: yaml.Node):
        rel_path = loader.construct_scalar(node)
        # normalize relative path against configs directory holding config.yaml
        if os.path.isabs(rel_path):
            include_path = rel_path
        else:
            # strip leading './'
            if rel_path.startswith('./'):
                rel_path = rel_path[2:]
            # avoid duplicated 'configs/' segment (configs/configs/...)
            if rel_path.startswith('configs/'):
                rel_path = rel_path[len('configs/'):]
            include_path = os.path.normpath(os.path.join(base_dir, rel_path))
        with open(include_path, 'r') as inc:
            return yaml.load(inc.read(), Loader=yaml.FullLoader)

    yaml.add_constructor('!include', _include_constructor, Loader=yaml.FullLoader)

with open(config_path,'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

runner = Generator(config)

# Always start a fresh run unless explicitly opted-in via env
version_dir = os.path.join(config["dataset"]["root"], config["dataset"]["version"])
attribute_json = os.path.join(version_dir, "attribute.json")
respect_existing = os.getenv("CARLA_NUSC_LOAD_EXISTING", "0") in ("1", "true", "True")
should_load = respect_existing and os.path.exists(attribute_json)

runner.generate_dataset(should_load)