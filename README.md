# Scan-To-Image-Predictor

스캔된 이미지를 기반으로 원본 이미지를 예측하는 딥러닝 모델 구현 프로젝트


## 프로젝트 주제: **Descanning**
- 스캔된 이미지를 원본 이미지와 같은 고화질 이미지로 추론하는 딥러닝 모델을 구현한다.


## **모델 후보**
1. **DnCNN**:  
   - 가우시안 노이즈를 제거하기 위해 설계된 모델.
   - **한계**: Score 30.0이 최대로 도출됨.

2. **CdNet**:  
   - 슈퍼 레졸루션 수행을 위한 트랜스포머 계열 모델.
   - **한계**: Score 4.9가 최대로 도출됨.

3. **HAT-S**:  
   - 고화질 변환을 목표로 하는 트랜스포머 계열 모델.
   - **특징**: Params 10M 이하와 학습 시간 제한을 만족하도록 경량화된 모델.

## **모델 선정 과정**

### DnCNN:
- 가우시안 노이즈 탐지를 위해 DnCNN 사용.
- **한계**: Score 30.0이 최대로 도출됨.


### CdNet:
- 이미지 딥러닝에서 좋은 성능을 보이는 트랜스포머 계열 모델.
- CdNet과 HAT 모델로 Super Resolution을 타겟으로 선정.
- **한계**: Score 4.9가 최대로 도출됨.

![Model Selection Chart 1](https://github.com/user-attachments/assets/316db9fc-8815-498e-a5ec-dfcbdb8535be)


### HAT-S:
- 경량화된 HAT-S 모델 선정.  
  - Params 수를 줄이고 학습 시간을 단축.
  - Competition 제한 사항(Params ≤ 10M) 충족.
- **결과**: Score 3.74 이하 도출.

![Model Selection Chart 2](https://github.com/user-attachments/assets/5ad2ccc5-fe3a-41b7-bdff-2c9b448ff30a)  
![Model Selection Chart 3](https://github.com/user-attachments/assets/dcbf70bc-6adc-4f9b-936c-fbca22fee6f2)


## **Task Pipeline**
- **BasicSR** 프레임워크 사용.


## **모델 결과**

### 1. Network Configuration:
```yaml
network_g:
  type: HAT
  upscale: 1
  in_chans: 3
  img_size: 512
  window_size: 8
  compress_ratio: 3
  squeeze_factor: 30
  conv_scale: 0.01
  overlap_ratio: 0.5
  img_range: 1.0
  depths: [6, 6, 6, 6]
  embed_dim: 96
  num_heads: [6, 6, 6, 6]
  mlp_ratio: 4.0
  upsampler: pixelshuffle
  resi_connection: 1conv
```

### 2. Loss Function:
- **Type**: L1Loss  
  - Loss Weight: 1.0  
  - Reduction: Mean  

### 3. Optimizer:
- **Adam**  
  - Learning Rate: 2e-4  
  - Weight Decay: 0  
  - Betas: [0.9, 0.99]  

## **모델 학습 결과**
![Training Results 1](https://github.com/user-attachments/assets/3fd0fe83-4529-4dcf-b52c-6445fe407723)  
![Training Results 2](https://github.com/user-attachments/assets/fcbed0b7-dd81-4657-8194-f8872e0e99a9)

## 참여자

- **고병후**: 모델 설계 및 구현, 최적화  
- **양동근**: 트레이닝 및 테스트  
- **Akhidjon**: 모델 설계  
