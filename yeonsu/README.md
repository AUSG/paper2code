# CIFAR-10 ResNet-18 학습 프로젝트

CIFAR-10 데이터셋을 사용하여 ResNet-18 모델을 학습하고, 특히 Cat과 Dog 클래스 간의 혼동 문제를 해결하기 위한 다양한 기법을 적용한 프로젝트입니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [설치 방법](#설치-방법)
- [데이터 준비](#데이터-준비)
- [사용 방법](#사용-방법)
- [실험 설정](#실험-설정)
- [프로젝트 구조](#프로젝트-구조)
- [실험 결과](#실험-결과)

## 🎯 프로젝트 개요

이 프로젝트는 CIFAR-10 데이터셋에서 ResNet-18 모델을 학습하며, 특히 유사한 특징을 가진 Cat과 Dog 클래스 간의 혼동 문제를 해결하기 위해 다음과 같은 기법들을 적용합니다:

- **다양한 손실 함수**: Focal Loss, Weighted Cross-Entropy Loss, Label Smoothing
- **고급 데이터 증강**: Cutout, Mixup, ColorJitter
- **체계적인 실험 관리**: 각 실험의 설정과 결과를 자동으로 저장

## ✨ 주요 기능

### 1. 다양한 손실 함수 지원

- **Cross-Entropy Loss**: 기본 손실 함수
- **Focal Loss**: 어려운 샘플에 집중하여 학습 (Cat/Dog 혼동 문제 해결에 효과적)
- **Weighted Cross-Entropy Loss**: 특정 클래스(Cat/Dog)에 높은 가중치 부여
- **Label Smoothing**: 과신뢰 방지 및 일반화 성능 향상

### 2. 데이터 증강 기법

- **Advanced Augmentation**: RandomHorizontalFlip, RandomCrop, ColorJitter
- **Cutout**: 이미지의 일부 영역을 제거하여 세부 특징에 의존하지 않도록 학습
- **Mixup**: 두 이미지를 혼합하여 결정 경계를 부드럽게 만듦

### 3. 실험 관리 시스템

- 각 실험마다 고유한 폴더 자동 생성
- 하이퍼파라미터 설정 자동 저장 (`config.json`)
- 학습 곡선 그래프 자동 생성 (`training_curves.png`)
- 학습 결과 CSV 저장 (`results.csv`)
- 최종 결과 요약 저장 (`summary.json`)

## 📦 설치 방법

### 패키지 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
- torch >= 2.0.0
- torchvision >= 0.15.0
- tqdm >= 4.65.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## 📁 데이터 준비

CIFAR-10 데이터셋을 자동으로 다운로드하고 준비합니다:

```bash
python download_cifar10.py
```

데이터는 `./data/cifar-10-batches-py/` 디렉토리에 저장됩니다.

## 🚀 사용 방법

### 기본 학습

가장 간단한 방법으로 학습을 시작합니다:

```bash
python train.py
```

### 고급 설정을 사용한 학습

Focal Loss와 고급 데이터 증강을 사용하여 학습:

```bash
python train.py \
    --epochs 150 \
    --lr 0.1 \
    --batch_size 128 \
    --loss_type focal \
    --focal_gamma 2.0 \
    --use_advanced_aug \
    --use_cutout \
    --cutout_length 16
```

### 제공된 스크립트 사용

프로젝트에는 여러 실험 설정이 포함된 스크립트가 제공됩니다:

```bash
# 기본 실험 (Cross-Entropy Loss)
bash scripts/baseline_train.sh

# 실험 1: Focal Loss + 고급 증강 + Cutout
bash scripts/exp1.sh

# 실험 2: Focal Loss + 고급 증강 + Cutout + Mixup
bash scripts/exp2.sh

# 실험 3: Weighted Loss + 고급 증강 + Cutout
bash scripts/exp3.sh
```

## ⚙️ 실험 설정

### 주요 하이퍼파라미터

#### 학습 관련
- `--epochs`: 학습 에폭 수 (기본값: 150)
- `--lr`: 초기 학습률 (기본값: 0.1)
- `--batch_size`: 배치 크기 (기본값: 128)
- `--momentum`: SGD momentum (기본값: 0.9)
- `--weight_decay`: Weight decay (기본값: 5e-4)

#### 학습률 스케줄러
- `--scheduler`: 스케줄러 타입 (`step`, `cosine`, `none`)
- `--step_size`: StepLR step_size (기본값: 30)
- `--gamma`: StepLR gamma (기본값: 0.1)

#### 손실 함수
- `--loss_type`: 손실 함수 타입 (`ce`, `focal`, `weighted`, `label_smooth`)
- `--focal_alpha`: Focal Loss alpha (기본값: 1.0)
- `--focal_gamma`: Focal Loss gamma (기본값: 2.0)
- `--class_weight`: Weighted Loss 클래스 가중치 (기본값: 2.0)
- `--label_smoothing`: Label Smoothing 값 (기본값: 0.1)

#### 데이터 증강
- `--use_advanced_aug`: 고급 데이터 증강 사용
- `--use_cutout`: Cutout 사용
- `--cutout_length`: Cutout 길이 (기본값: 16)
- `--use_mixup`: Mixup 사용
- `--mixup_alpha`: Mixup alpha (기본값: 1.0)

### 추천 설정 조합

#### 조합 1: 기본 개선 (추천)
```bash
python train.py \
    --use_advanced_aug \
    --use_cutout \
    --cutout_length 16 \
    --loss_type focal \
    --focal_gamma 2.0
```

#### 조합 2: 강력한 개선
```bash
python train.py \
    --use_advanced_aug \
    --use_cutout \
    --cutout_length 16 \
    --loss_type focal \
    --focal_gamma 2.5 \
    --use_mixup \
    --mixup_alpha 0.8
```

#### 조합 3: 극대화
```bash
python train.py \
    --use_advanced_aug \
    --use_cutout \
    --cutout_length 16 \
    --loss_type weighted \
    --class_weight 2.0 \
    --use_mixup \
    --mixup_alpha 1.0
```

## 📂 프로젝트 구조

```
yeonsu/
├── README.md                 # 이 파일
├── requirements.txt          # 의존성 패키지 목록
├── train.py                  # 메인 학습 스크립트
├── dataset.py                # 데이터 로더
├── losses.py                 # 손실 함수 구현
├── augmentation.py           # 데이터 증강 구현
├── download_cifar10.py       # CIFAR-10 다운로드 스크립트
├── models/
│   └── resnet18.py          # ResNet-18 모델 정의
├── scripts/
│   ├── baseline_train.sh    # 기본 학습 스크립트
│   ├── exp1.sh              # 실험 1 스크립트
│   ├── exp2.sh              # 실험 2 스크립트
│   └── exp3.sh              # 실험 3 스크립트
├── data/                     # 데이터 디렉토리
│   └── cifar-10-batches-py/
├── experiments/              # 실험 결과 저장 디렉토리
│   ├── ce_20260106_133608/
│   │   ├── config.json      # 실험 설정
│   │   ├── summary.json     # 결과 요약
│   │   ├── results.csv      # 학습 결과
│   │   ├── training_curves.png  # 학습 곡선
│   │   └── checkpoints/     # 모델 체크포인트
│   └── ...
└── augmentation_examples/    # 증강 기법 예제 이미지
    ├── cutout_example.png
    └── mixup_example.png
```

## 📊 실험 결과

각 실험의 결과는 `experiments/` 디렉토리에 저장됩니다. 각 실험 폴더에는 다음 파일들이 포함됩니다:

- `config.json`: 사용된 하이퍼파라미터 설정
- `summary.json`: 최고 정확도, 최종 정확도, 학습 시간 등 요약 정보
- `results.csv`: 에폭별 학습/테스트 손실 및 정확도
- `training_curves.png`: 학습 곡선 시각화
- `checkpoints/`: 모델 체크포인트 파일들

### 예시 결과

실험 예시:
- **기본 (CE)**: 최고 정확도 ~94%
- **Focal Loss + 증강**: 최고 정확도 ~95%+
- **Focal Loss + 증강 + Mixup**: 최고 정확도 ~94+
