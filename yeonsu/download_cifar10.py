#!/usr/bin/env python3
"""
CIFAR-10 데이터셋 다운로드 스크립트
torchvision을 사용하여 CIFAR-10 데이터를 다운로드합니다.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os

def download_cifar10(data_dir='./data'):
    """
    CIFAR-10 데이터셋을 다운로드합니다.
    
    Args:
        data_dir: 데이터를 저장할 디렉토리 경로
    """
    # 데이터 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"CIFAR-10 데이터를 {data_dir} 디렉토리에 다운로드합니다...")
    
    # 데이터 변환 (다운로드를 위한 기본 변환)
    transform = transforms.ToTensor()
    
    # CIFAR-10 학습 데이터셋 다운로드
    print("학습 데이터셋 다운로드 중...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    print(f"학습 데이터셋 다운로드 완료: {len(trainset)}개 이미지")
    
    # CIFAR-10 테스트 데이터셋 다운로드
    print("테스트 데이터셋 다운로드 중...")
    testset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    print(f"테스트 데이터셋 다운로드 완료: {len(testset)}개 이미지")
    
    # 클래스 이름 출력
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    print("\nCIFAR-10 클래스:")
    for i, class_name in enumerate(classes):
        print(f"  {i}: {class_name}")
    
    print(f"\n다운로드 완료! 데이터는 {data_dir}/cifar-10-batches-py/ 디렉토리에 저장되었습니다.")
    return trainset, testset

if __name__ == "__main__":
    # 현재 스크립트가 있는 디렉토리에 data 폴더 생성
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    try:
        trainset, testset = download_cifar10(data_dir)
        print("\n성공적으로 CIFAR-10 데이터를 다운로드했습니다!")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

