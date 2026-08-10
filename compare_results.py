import pandas as pd

for name, path in [
    ('ResNet18', 'results/kfold_results_resnet18.csv'),
    ('EfficientNet-B0', 'results/kfold_results_efficientnet_b0.csv'),
]:
    df = pd.read_csv(path)
    print(f'\n=== {name} ===')
    print(f"  MAE  : {df['MAE'].mean():.4f} +/- {df['MAE'].std():.4f}")
    print(f"  RMSE : {df['RMSE'].mean():.4f} +/- {df['RMSE'].std():.4f}")
    print(f"  R2   : {df['R2'].mean():.4f} +/- {df['R2'].std():.4f}")
