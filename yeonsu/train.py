import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import os
import time
import json
import csv
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.resnet18 import resnet18
from dataset import get_cifar10_dataloaders
from losses import FocalLoss, LabelSmoothingCrossEntropy, WeightedCrossEntropyLoss, get_cat_dog_focused_weights
from augmentation import Mixup, mixup_criterion


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_mixup=False, mixup_alpha=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    mixup_fn = Mixup(alpha=mixup_alpha) if use_mixup else None

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_mixup and mixup_fn:
            inputs, targets_a, targets_b, lam = mixup_fn((inputs, targets))
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if use_mixup and mixup_fn:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if use_mixup:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item() + 
                       (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def save_checkpoint(model, optimizer, scheduler, epoch, acc, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': acc,
    }
    torch.save(checkpoint, filepath)
    print(f'체크포인트 저장: {filepath}')


def load_checkpoint(model, optimizer, scheduler, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    acc = checkpoint['accuracy']
    print(f'체크포인트 로드: {filepath} (에폭 {epoch}, 정확도 {acc:.2f}%)')
    return epoch, acc


def create_experiment_folder(config, base_dir='./experiments'):
    """실험 폴더 생성 및 하이퍼파라미터 저장"""
    # 실험 이름 생성
    exp_name_parts = []
    
    # 손실 함수
    if config['loss_type'] == 'focal':
        exp_name_parts.append(f"focal_g{config['focal_gamma']}")
    elif config['loss_type'] == 'weighted':
        exp_name_parts.append(f"weighted_w{config['class_weight']}")
    elif config['loss_type'] == 'label_smooth':
        exp_name_parts.append(f"labelsmooth_s{config['label_smoothing']}")
    else:
        exp_name_parts.append("ce")
    
    # 데이터 증강
    aug_parts = []
    if config['use_advanced_aug']:
        aug_parts.append("adv_aug")
    if config['use_cutout']:
        aug_parts.append(f"cutout{config['cutout_length']}")
    if config['use_mixup']:
        aug_parts.append(f"mixup{config['mixup_alpha']}")
    
    if aug_parts:
        exp_name_parts.append("_".join(aug_parts))
    
    # 타임스탬프 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = "_".join(exp_name_parts) + f"_{timestamp}"
    
    # 실험 폴더 생성
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 하이퍼파라미터 저장
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f'실험 폴더 생성: {exp_dir}')
    print(f'하이퍼파라미터 저장: {config_path}')
    
    return exp_dir


def save_results(exp_dir, train_losses, train_accs, test_losses, test_accs):
    """학습 결과를 CSV로 저장"""
    results_path = os.path.join(exp_dir, 'results.csv')
    
    with open(results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                f'{train_losses[epoch]:.6f}',
                f'{train_accs[epoch]:.2f}',
                f'{test_losses[epoch]:.6f}',
                f'{test_accs[epoch]:.2f}'
            ])
    
    print(f'학습 결과 저장: {results_path}')


def save_summary(exp_dir, best_acc, final_acc, total_time):
    """최종 결과 요약 저장"""
    summary = {
        'best_test_accuracy': best_acc,
        'final_test_accuracy': final_acc,
        'total_training_time_seconds': total_time,
        'completed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = os.path.join(exp_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f'결과 요약 저장: {summary_path}')


def plot_training_curves(exp_dir, train_losses, train_accs, test_losses, test_accs):
    """학습 곡선 그래프 저장"""
    epochs = range(1, len(train_losses) + 1)
    
    # Figure 생성 (2x2 subplot)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Train/Test Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Train/Test Accuracy
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss 비교 (확대) - 마지막 절반 에폭의 범위로 확대
    axes[1, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
    axes[1, 0].plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Loss (Zoomed)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    # 마지막 절반 에폭의 범위로 확대하여 세밀한 변화 관찰
    if len(train_losses) > 0:
        half_point = len(train_losses) // 2
        zoom_losses = train_losses[half_point:] + test_losses[half_point:]
        if len(zoom_losses) > 0:
            min_loss = min(zoom_losses)
            max_loss = max(zoom_losses)
            margin = (max_loss - min_loss) * 0.15
            axes[1, 0].set_ylim([max(0, min_loss - margin), max_loss + margin])
    
    # 4. Accuracy 비교 (확대) - 마지막 절반 에폭의 범위로 확대
    axes[1, 1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2, alpha=0.7)
    axes[1, 1].plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Accuracy (Zoomed)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    # 마지막 절반 에폭의 범위로 확대하여 세밀한 변화 관찰
    if len(train_accs) > 0:
        half_point = len(train_accs) // 2
        zoom_accs = train_accs[half_point:] + test_accs[half_point:]
        if len(zoom_accs) > 0:
            min_acc = min(zoom_accs)
            max_acc = max(zoom_accs)
            margin = (max_acc - min_acc) * 0.15
            axes[1, 1].set_ylim([max(0, min_acc - margin), min(100, max_acc + margin)])
    
    plt.tight_layout()
    
    # 그래프 저장
    plot_path = os.path.join(exp_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'학습 곡선 그래프 저장: {plot_path}')


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CIFAR-10 ResNet-18 학습')
    
    # 데이터 관련
    parser.add_argument('--data_dir', type=str, default='./data', help='데이터 디렉토리')
    parser.add_argument('--batch_size', type=int, default=128, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로더 워커 수')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=150, help='학습 에폭 수')
    parser.add_argument('--lr', type=float, default=0.1, help='초기 학습률')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    
    # 스케줄러
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'none'], help='학습률 스케줄러')
    parser.add_argument('--step_size', type=int, default=30, help='StepLR step_size')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR gamma')
    
    # 저장 관련
    parser.add_argument('--save_freq', type=int, default=10, help='체크포인트 저장 주기')
    
    # 데이터 증강
    parser.add_argument('--use_advanced_aug', action='store_true', help='고급 데이터 증강 사용')
    parser.add_argument('--use_cutout', action='store_true', help='Cutout 사용')
    parser.add_argument('--cutout_length', type=int, default=16, help='Cutout 길이')
    parser.add_argument('--use_mixup', action='store_true', help='Mixup 사용')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Mixup alpha')
    
    # 손실 함수
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal', 'weighted', 'label_smooth'], help='손실 함수 타입')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal Loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma')
    parser.add_argument('--class_weight', type=float, default=2.0, help='Weighted Loss 클래스 가중치')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label Smoothing 값')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'epochs': args.epochs,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'save_dir': './checkpoints',
        'save_freq': args.save_freq,
        
        'use_advanced_aug': args.use_advanced_aug,
        'use_cutout': args.use_cutout,
        'cutout_length': args.cutout_length,
        'use_mixup': args.use_mixup,
        'mixup_alpha': args.mixup_alpha,
        
        'loss_type': args.loss_type,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'class_weight': args.class_weight,
        'label_smoothing': args.label_smoothing,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 실험 폴더 생성
    exp_dir = create_experiment_folder(config, base_dir='./experiments')
    config['save_dir'] = os.path.join(exp_dir, 'checkpoints')
    
    train_loader, test_loader, classes = get_cifar10_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_advanced_aug=config['use_advanced_aug'],
        use_cutout=config['use_cutout'],
        cutout_length=config['cutout_length']
    )
    
    model = resnet18(num_classes=10).to(device)
    
    if config['loss_type'] == 'focal':
        criterion = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma']
        ).to(device)
    elif config['loss_type'] == 'weighted':
        class_weights = get_cat_dog_focused_weights(
            num_classes=10,
            weight=config['class_weight']
        )
        criterion = WeightedCrossEntropyLoss(
            class_weights=class_weights
        ).to(device)
    elif config['loss_type'] == 'label_smooth':
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config['label_smoothing'],
            num_classes=10
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    
    if config['use_mixup']:
        print(f'Mixup 사용: alpha={config["mixup_alpha"]}')
    if config['use_advanced_aug']:
        print(f'고급 데이터 증강 사용: Cutout={config["use_cutout"]}')
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    if config['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    else:
        scheduler = None
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    best_acc = 0.0
    total_start_time = time.time()
    
    print(f'\n학습 시작 (총 {config["epochs"]} 에폭)...')
    print('=' * 60)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(config['epochs']):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_mixup=config['use_mixup'],
            mixup_alpha=config['mixup_alpha']
        )
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config['lr']
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        epoch_time = time.time() - start_time
        print(f'\nEpoch {epoch+1}/{config["epochs"]} ({epoch_time:.1f}초)')
        print(f'학습 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'테스트 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        print('-' * 60)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(config['save_dir'], 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, test_acc, best_path)
        
        if (epoch + 1) % config['save_freq'] == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, test_acc, checkpoint_path)
    
    total_time = time.time() - total_start_time
    final_acc = test_accs[-1] if len(test_accs) > 0 else 0.0
    
    # 결과 저장
    save_results(exp_dir, train_losses, train_accs, test_losses, test_accs)
    save_summary(exp_dir, best_acc, final_acc, total_time)
    plot_training_curves(exp_dir, train_losses, train_accs, test_losses, test_accs)
    
    print('\n' + '=' * 60)
    print('학습 완료!')
    print(f'실험 폴더: {exp_dir}')
    print(f'최고 테스트 정확도: {best_acc:.2f}%')
    print(f'최종 테스트 정확도: {final_acc:.2f}%')
    print(f'총 학습 시간: {total_time/60:.1f}분')
    print('=' * 60)


if __name__ == '__main__':
    main()

