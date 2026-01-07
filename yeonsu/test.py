"""
ResNet-18 모델 테스트/평가 스크립트
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from models.resnet18 import resnet18
from dataset import get_cifar10_dataloaders


def test_model(model, test_loader, device, classes=None, save_confusion_matrix=False):
    """모델 테스트 및 상세 통계 출력"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0
    correct = 0
    total = 0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    
    # Confusion matrix를 위한 예측값과 실제값 저장
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Confusion matrix용 데이터 수집
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 클래스별 정확도 계산
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 전체 통계
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    print('\n' + '=' * 60)
    print('테스트 결과')
    print('=' * 60)
    print(f'전체 테스트 손실: {test_loss:.4f}')
    print(f'전체 테스트 정확도: {test_acc:.2f}% ({correct}/{total})')
    print('=' * 60)
    
    # 클래스별 정확도
    if classes:
        print('\n클래스별 정확도:')
        print('-' * 60)
        for i in range(10):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f'{classes[i]:10s}: {acc:6.2f}% ({int(class_correct[i]):5d}/{int(class_total[i]):5d})')
            else:
                print(f'{classes[i]:10s}: N/A')
        print('-' * 60)
    
    # Confusion Matrix 계산 및 출력
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    cm = confusion_matrix(all_targets, all_preds)
    
    print('\n' + '=' * 60)
    print('Confusion Matrix')
    print('=' * 60)
    print_confusion_matrix(cm, classes)
    
    # Classification Report
    if classes:
        print('\n' + '=' * 60)
        print('Classification Report')
        print('=' * 60)
        report = classification_report(
            all_targets, all_preds, 
            target_names=classes,
            digits=4
        )
        print(report)
    
    return test_loss, test_acc, cm


def print_confusion_matrix(cm, classes=None):
    """Confusion Matrix를 텍스트로 출력"""
    if classes is None:
        classes = [f'Class {i}' for i in range(len(cm))]
    
    # 헤더 출력
    print(f'\n{"실제/예측":>12}', end='')
    for cls in classes:
        print(f'{cls[:6]:>8}', end='')
    print()
    print('-' * (12 + 8 * len(classes)))
    
    # 행 출력
    for i, cls in enumerate(classes):
        print(f'{cls:>12}', end='')
        for j in range(len(classes)):
            print(f'{cm[i, j]:>8}', end='')
        print(f'  (정확도: {100*cm[i,i]/cm[i].sum():.1f}%)')
    print('-' * (12 + 8 * len(classes)))


def plot_confusion_matrix(cm, classes=None, save_path='confusion_matrix.png', figsize=(10, 8)):
    """Confusion Matrix를 시각화하여 저장"""
    if classes is None:
        classes = [f'Class {i}' for i in range(len(cm))]
    
    # 정규화된 confusion matrix 계산 (비율로)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Figure 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 원본 Confusion Matrix (개수)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Count)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.tick_params(axis='both', labelsize=9)
    
    # 정규화된 Confusion Matrix (비율)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion Matrix 이미지 저장 완료: {save_path}')


def load_model(checkpoint_path, device, num_classes=10):
    """체크포인트에서 모델 로드"""
    print(f'체크포인트 로드: {checkpoint_path}')
    
    # 모델 생성
    model = resnet18(num_classes=num_classes).to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint and 'accuracy' in checkpoint:
            print(f'체크포인트 정보: 에폭 {checkpoint["epoch"]}, 정확도 {checkpoint["accuracy"]:.2f}%')
    else:
        # state_dict만 저장된 경우
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='ResNet-18 CIFAR-10 테스트')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='체크포인트 파일 경로')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='데이터 디렉토리 경로')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='데이터 로딩 워커 수')
    parser.add_argument('--save_cm', action='store_true',
                        help='Confusion Matrix 이미지 저장')
    parser.add_argument('--cm_path', type=str, default='confusion_matrix.png',
                        help='Confusion Matrix 저장 경로')
    
    args = parser.parse_args()
    
    # 체크포인트 파일 확인
    if not os.path.exists(args.checkpoint):
        print(f'오류: 체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}')
        return
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 디바이스: {device}')
    
    # 모델 로드
    model = load_model(args.checkpoint, device)
    
    # 데이터 로더
    print('\n데이터 로더 준비 중...')
    _, test_loader, classes = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 테스트 실행
    test_loss, test_acc, cm = test_model(
        model, test_loader, device, classes, 
        save_confusion_matrix=args.save_cm
    )
    
    if args.save_cm:
        plot_confusion_matrix(cm, classes, save_path=args.cm_path)
    
    print(f'\n최종 테스트 정확도: {test_acc:.2f}%')


if __name__ == '__main__':
    main()

