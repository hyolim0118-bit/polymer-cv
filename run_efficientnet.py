import importlib.util
from pathlib import Path
import shutil

spec = importlib.util.spec_from_file_location(
    'cross_validation', 'training/training real/cross_validation.py'
)
cross_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross_validation)

from datasets.dataset import load_config

config = load_config(Path('training/config.yaml'))
config['model']['name'] = 'efficientnet_b0'
cross_validation.run_kfold(config)
shutil.copy('results/kfold_results.csv', 'results/kfold_results_efficientnet_b0.csv')
print('EfficientNet-B0 5-fold 완료')
