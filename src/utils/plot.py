import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def create_training_plots(log_file, save_dir='plots'):
    """
    创建训练过程的可视化图表
    Args:
        log_file: 包含训练数据的日志文件
        save_dir: 保存图表的目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 读取日志数据
    with open(log_file, 'r') as f:
        log_data = json.load(f)

    # 提取训练数据
    epochs = log_data['epochs']
    train_losses = log_data['train_losses']
    val_losses = log_data['val_losses']
    train_accs = log_data['train_accs']
    test_accs = log_data['test_accs']
    folds = log_data['folds']

    for fold in range(folds):
        val_losses[fold].append(0.0)

    # 绘制训练/验证损失曲线
    plt.figure(figsize=(12, 6))
    for fold in range(folds):
        plt.plot(epochs[fold], train_losses[fold],
                 label=f'Fold {fold+1} Train Loss')
        plt.plot(epochs[fold], val_losses[fold],
                 label=f'Fold {fold+1} Val Loss', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_curves.png', dpi=300)
    plt.close()

    # 绘制训练/测试准确率曲线
    plt.figure(figsize=(12, 6))
    for fold in range(folds):
        epochs_for_acc_plot = [
            (i + 1) * 10 for i in range(len(test_accs[fold]))]
        plt.plot(epochs_for_acc_plot, train_accs[fold],
                 label=f'Fold {fold+1} Train Acc')
        plt.plot(epochs_for_acc_plot, test_accs[fold],
                 label=f'Fold {fold+1} Test Acc', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_curves.png', dpi=300)
    plt.close()

    # 绘制每折最终性能条形图
    final_train_accs = [train_accs[i][-1] for i in range(folds)]
    final_test_accs = [test_accs[i][-1] for i in range(folds)]

    plt.figure(figsize=(10, 6))
    x = np.arange(folds)
    width = 0.35

    plt.bar(x - width/2, final_train_accs, width, label='Train Accuracy')
    plt.bar(x + width/2, final_test_accs, width, label='Test Accuracy')

    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Final Accuracy by Fold')
    plt.xticks(x, [f'Fold {i+1}' for i in range(folds)])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/final_accuracy_by_fold.png', dpi=300)
    plt.close()

    # 计算平均性能
    avg_train_acc = np.mean(final_train_accs)
    avg_test_acc = np.mean(final_test_accs)
    std_train_acc = np.std(final_train_accs)
    std_test_acc = np.std(final_test_accs)

    # 保存平均性能数据
    performance_summary = {
        'avg_train_acc': float(avg_train_acc),
        'avg_test_acc': float(avg_test_acc),
        'std_train_acc': float(std_train_acc),
        'std_test_acc': float(std_test_acc)
    }

    with open(f'{save_dir}/performance_summary.json', 'w') as f:
        json.dump(performance_summary, f, indent=4)

    print(f"Average Train Accuracy: {avg_train_acc:.4f} ± {std_train_acc:.4f}")
    print(f"Average Test Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")

    return performance_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Path to the log file containing training data')
    parser.add_argument('--save_dir', type=str, default='plots',
                        help='Directory to save plots')
    args = parser.parse_args()

    file_name_with_ext = os.path.basename(args.log_file)
    file_name_without_ext, _ = os.path.splitext(file_name_with_ext)

    final_plot_save_dir = os.path.join(args.save_dir, file_name_without_ext)

    os.makedirs(final_plot_save_dir, exist_ok=True)
    create_training_plots(args.log_file, final_plot_save_dir)
