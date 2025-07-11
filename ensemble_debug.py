import pandas as pd
import os

# Extracted results from the provided output
results_data = [
    # cifar10_resnet20
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Original", "attack": "FGSM", "clean_acc": 92.0, "adv_acc": 49.0, "dropped_percentage": 46.7, "attack_time": 0.5, "num_samples": 100},
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Original", "attack": "PGD", "clean_acc": 92.0, "adv_acc": 44.0, "dropped_percentage": 52.2, "attack_time": 6.4, "num_samples": 100},
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Original", "attack": "BIM", "clean_acc": 92.0, "adv_acc": 44.0, "dropped_percentage": 52.2, "attack_time": 4.8, "num_samples": 100},
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Original", "attack": "MEA", "clean_acc": 92.0, "adv_acc": 45.0, "dropped_percentage": 51.1, "attack_time": 12.2, "num_samples": 100},
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Original", "attack": "WEA", "clean_acc": 92.0, "adv_acc": 45.0, "dropped_percentage": 51.1, "attack_time": 12.3, "num_samples": 100},
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "FGSM", "clean_acc": 82.0, "adv_acc": 43.0, "dropped_percentage": 47.6, "attack_time": 0.5, "num_samples": 100},
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "PGD", "clean_acc": 82.0, "adv_acc": 37.0, "dropped_percentage": 54.9, "attack_time": 7.3, "num_samples": 100},
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "BIM", "clean_acc": 82.0, "adv_acc": 37.0, "dropped_percentage": 54.9, "attack_time": 5.0, "num_samples": 100},
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "MEA", "clean_acc": 82.0, "adv_acc": 39.0, "dropped_percentage": 52.4, "attack_time": 14.3, "num_samples": 100},
    {"model_name": "cifar10_resnet20", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "WEA", "clean_acc": 82.0, "adv_acc": 40.0, "dropped_percentage": 51.2, "attack_time": 12.5, "num_samples": 100},
    
    # cifar10_resnet32
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Original", "attack": "FGSM", "clean_acc": 95.0, "adv_acc": 47.0, "dropped_percentage": 50.5, "attack_time": 0.9, "num_samples": 100},
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Original", "attack": "PGD", "clean_acc": 95.0, "adv_acc": 42.0, "dropped_percentage": 55.8, "attack_time": 12.3, "num_samples": 100},
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Original", "attack": "BIM", "clean_acc": 95.0, "adv_acc": 42.0, "dropped_percentage": 55.8, "attack_time": 8.0, "num_samples": 100},
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Original", "attack": "MEA", "clean_acc": 95.0, "adv_acc": 42.0, "dropped_percentage": 55.8, "attack_time": 20.8, "num_samples": 100},
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Original", "attack": "WEA", "clean_acc": 95.0, "adv_acc": 43.0, "dropped_percentage": 54.7, "attack_time": 21.7, "num_samples": 100},
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "FGSM", "clean_acc": 79.0, "adv_acc": 42.0, "dropped_percentage": 46.8, "attack_time": 1.9, "num_samples": 100},
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "PGD", "clean_acc": 79.0, "adv_acc": 40.0, "dropped_percentage": 49.4, "attack_time": 12.0, "num_samples": 100},
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "BIM", "clean_acc": 79.0, "adv_acc": 40.0, "dropped_percentage": 49.4, "attack_time": 7.6, "num_samples": 100},
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "MEA", "clean_acc": 79.0, "adv_acc": 40.0, "dropped_percentage": 49.4, "attack_time": 23.2, "num_samples": 100},
    {"model_name": "cifar10_resnet32", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "WEA", "clean_acc": 79.0, "adv_acc": 40.0, "dropped_percentage": 49.4, "attack_time": 25.0, "num_samples": 100},
    
    # cifar10_resnet44
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Original", "attack": "FGSM", "clean_acc": 93.0, "adv_acc": 52.0, "dropped_percentage": 44.1, "attack_time": 1.1, "num_samples": 100},
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Original", "attack": "PGD", "clean_acc": 93.0, "adv_acc": 49.0, "dropped_percentage": 47.3, "attack_time": 16.2, "num_samples": 100},
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Original", "attack": "BIM", "clean_acc": 93.0, "adv_acc": 49.0, "dropped_percentage": 47.3, "attack_time": 13.3, "num_samples": 100},
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Original", "attack": "MEA", "clean_acc": 93.0, "adv_acc": 49.0, "dropped_percentage": 47.3, "attack_time": 28.0, "num_samples": 100},
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Original", "attack": "WEA", "clean_acc": 93.0, "adv_acc": 50.0, "dropped_percentage": 46.2, "attack_time": 27.6, "num_samples": 100},
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "FGSM", "clean_acc": 79.0, "adv_acc": 45.0, "dropped_percentage": 43.0, "attack_time": 1.3, "num_samples": 100},
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "PGD", "clean_acc": 79.0, "adv_acc": 40.0, "dropped_percentage": 49.4, "attack_time": 15.9, "num_samples": 100},
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "BIM", "clean_acc": 79.0, "adv_acc": 40.0, "dropped_percentage": 49.4, "attack_time": 11.6, "num_samples": 100},
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "MEA", "clean_acc": 79.0, "adv_acc": 42.0, "dropped_percentage": 46.8, "attack_time": 76.4, "num_samples": 100},
    {"model_name": "cifar10_resnet44", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "WEA", "clean_acc": 79.0, "adv_acc": 42.0, "dropped_percentage": 46.8, "attack_time": 60.8, "num_samples": 100},
    
    # cifar10_resnet56
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Original", "attack": "FGSM", "clean_acc": 96.0, "adv_acc": 56.0, "dropped_percentage": 41.7, "attack_time": 3.2, "num_samples": 100},
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Original", "attack": "PGD", "clean_acc": 96.0, "adv_acc": 53.0, "dropped_percentage": 44.8, "attack_time": 33.7, "num_samples": 100},
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Original", "attack": "BIM", "clean_acc": 96.0, "adv_acc": 53.0, "dropped_percentage": 44.8, "attack_time": 14.3, "num_samples": 100},
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Original", "attack": "MEA", "clean_acc": 96.0, "adv_acc": 55.0, "dropped_percentage": 42.7, "attack_time": 34.6, "num_samples": 100},
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Original", "attack": "WEA", "clean_acc": 96.0, "adv_acc": 55.0, "dropped_percentage": 42.7, "attack_time": 34.8, "num_samples": 100},
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "FGSM", "clean_acc": 83.0, "adv_acc": 52.0, "dropped_percentage": 37.3, "attack_time": 1.5, "num_samples": 100},
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "PGD", "clean_acc": 83.0, "adv_acc": 49.0, "dropped_percentage": 41.0, "attack_time": 20.6, "num_samples": 100},
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "BIM", "clean_acc": 83.0, "adv_acc": 49.0, "dropped_percentage": 41.0, "attack_time": 14.2, "num_samples": 100},
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "MEA", "clean_acc": 83.0, "adv_acc": 49.0, "dropped_percentage": 41.0, "attack_time": 34.3, "num_samples": 100},
    {"model_name": "cifar10_resnet56", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "WEA", "clean_acc": 83.0, "adv_acc": 50.0, "dropped_percentage": 39.8, "attack_time": 38.6, "num_samples": 100},
    
    # cifar10_mobilenetv2_x1_4
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Original", "attack": "FGSM", "clean_acc": 91.0, "adv_acc": 53.0, "dropped_percentage": 41.8, "attack_time": 3.7, "num_samples": 100},
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Original", "attack": "PGD", "clean_acc": 91.0, "adv_acc": 52.0, "dropped_percentage": 42.9, "attack_time": 50.6, "num_samples": 100},
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Original", "attack": "BIM", "clean_acc": 91.0, "adv_acc": 52.0, "dropped_percentage": 42.9, "attack_time": 34.6, "num_samples": 100},
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Original", "attack": "MEA", "clean_acc": 91.0, "adv_acc": 52.0, "dropped_percentage": 42.9, "attack_time": 93.7, "num_samples": 100},
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Original", "attack": "WEA", "clean_acc": 91.0, "adv_acc": 52.0, "dropped_percentage": 42.9, "attack_time": 86.2, "num_samples": 100},
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "FGSM", "clean_acc": 88.0, "adv_acc": 57.0, "dropped_percentage": 35.2, "attack_time": 3.3, "num_samples": 100},
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "PGD", "clean_acc": 88.0, "adv_acc": 54.0, "dropped_percentage": 38.6, "attack_time": 52.4, "num_samples": 100},
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "BIM", "clean_acc": 88.0, "adv_acc": 54.0, "dropped_percentage": 38.6, "attack_time": 31.3, "num_samples": 100},
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "MEA", "clean_acc": 88.0, "adv_acc": 55.0, "dropped_percentage": 37.5, "attack_time": 90.1, "num_samples": 100},
    {"model_name": "cifar10_mobilenetv2_x1_4", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "WEA", "clean_acc": 88.0, "adv_acc": 56.0, "dropped_percentage": 36.4, "attack_time": 83.3, "num_samples": 100},
    
    # cifar10_vgg16_bn
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "FGSM", "clean_acc": 95.0, "adv_acc": 59.0, "dropped_percentage": 37.9, "attack_time": 4.0, "num_samples": 100},
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "PGD", "clean_acc": 95.0, "adv_acc": 57.0, "dropped_percentage": 40.0, "attack_time": 45.2, "num_samples": 100},
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "BIM", "clean_acc": 95.0, "adv_acc": 57.0, "dropped_percentage": 40.0, "attack_time": 31.3, "num_samples": 100},
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "MEA", "clean_acc": 95.0, "adv_acc": 57.0, "dropped_percentage": 40.0, "attack_time": 78.3, "num_samples": 100},
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "WEA", "clean_acc": 95.0, "adv_acc": 57.0, "dropped_percentage": 40.0, "attack_time": 76.3, "num_samples": 100},
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "FGSM", "clean_acc": 84.0, "adv_acc": 39.0, "dropped_percentage": 53.6, "attack_time": 2.7, "num_samples": 100},
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "PGD", "clean_acc": 84.0, "adv_acc": 38.0, "dropped_percentage": 54.8, "attack_time": 38.9, "num_samples": 100},
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "BIM", "clean_acc": 84.0, "adv_acc": 38.0, "dropped_percentage": 54.8, "attack_time": 39.1, "num_samples": 100},
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "MEA", "clean_acc": 84.0, "adv_acc": 38.0, "dropped_percentage": 54.8, "attack_time": 69.9, "num_samples": 100},
    {"model_name": "cifar10_vgg16_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "WEA", "clean_acc": 84.0, "adv_acc": 39.0, "dropped_percentage": 53.6, "attack_time": 72.5, "num_samples": 100},
    
    # cifar10_vgg11_bn
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "FGSM", "clean_acc": 94.0, "adv_acc": 50.0, "dropped_percentage": 46.8, "attack_time": 1.5, "num_samples": 100},
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "PGD", "clean_acc": 94.0, "adv_acc": 47.0, "dropped_percentage": 50.0, "attack_time": 23.5, "num_samples": 100},
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "BIM", "clean_acc": 94.0, "adv_acc": 47.0, "dropped_percentage": 50.0, "attack_time": 18.8, "num_samples": 100},
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "MEA", "clean_acc": 94.0, "adv_acc": 49.0, "dropped_percentage": 47.9, "attack_time": 38.7, "num_samples": 100},
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Original", "attack": "WEA", "clean_acc": 94.0, "adv_acc": 49.0, "dropped_percentage": 47.9, "attack_time": 45.4, "num_samples": 100},
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "FGSM", "clean_acc": 83.0, "adv_acc": 45.0, "dropped_percentage": 45.8, "attack_time": 2.7, "num_samples": 100},
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "PGD", "clean_acc": 83.0, "adv_acc": 44.0, "dropped_percentage": 47.0, "attack_time": 21.5, "num_samples": 100},
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "BIM", "clean_acc": 83.0, "adv_acc": 44.0, "dropped_percentage": 47.0, "attack_time": 17.3, "num_samples": 100},
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "MEA", "clean_acc": 83.0, "adv_acc": 44.0, "dropped_percentage": 47.0, "attack_time": 39.6, "num_samples": 100},
    {"model_name": "cifar10_vgg11_bn", "dataset": "CIFAR10", "model_type": "Distilled", "attack": "WEA", "clean_acc": 83.0, "adv_acc": 44.0, "dropped_percentage": 47.0, "attack_time": 46.2, "num_samples": 100}
]

def main():
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Create results directory if it doesn't exist
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(results_dir, "cifar10_attack_results.csv")
    df.to_csv(output_file, index=False)
    
    print(f"CSV created with {len(df)} rows")
    print(f"Saved to: {output_file}")
    
    # Display summary
    print(f"\nDataset breakdown:")
    print(df['dataset'].value_counts())
    
    print(f"\nModel type breakdown:")
    print(df['model_type'].value_counts())
    
    print(f"\nAttack breakdown:")
    print(df['attack'].value_counts())
    
    print(f"\nSample of the data:")
    print(df.head(10))
    
    # Show average dropped percentages by attack
    print(f"\nAverage Dropped % by Attack:")
    attack_avg = df.groupby('attack')['dropped_percentage'].mean().round(1)
    for attack, avg_drop in attack_avg.items():
        print(f"  {attack}: {avg_drop}%")
    
    # Show average dropped percentages by model type
    print(f"\nAverage Dropped % by Model Type:")
    model_avg = df.groupby('model_type')['dropped_percentage'].mean().round(1)
    for model_type, avg_drop in model_avg.items():
        print(f"  {model_type}: {avg_drop}%")

if __name__ == "__main__":
    main()