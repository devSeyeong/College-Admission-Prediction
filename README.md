# 대학 합격 예측 모델 (University Admission Prediction Model)

이 프로젝트는 학생들의 GRE 점수, GPA, 학교 랭킹을 기반으로 대학 합격 여부를 예측하는 이진 분류 모델을 구축하는 것입니다. TensorFlow와 Keras를 사용하여 신경망 모델을 정의하고 학습시킨 후, 새로운 데이터에 대해 합격 예측을 수행합니다.

## 프로젝트 개요
- **목표**: 주어진 학생의 특성(GRE 점수, GPA, 학교 랭킹)을 바탕으로 대학에 합격할 가능성을 예측하는 이진 분류 문제입니다.
- **입력 데이터**: 학생의 **GRE 점수**, **GPA**, **학교 랭킹**
- **출력 데이터**: 학생의 대학 **합격 여부**(0 = 불합격, 1 = 합격)

## 사용한 라이브러리
- `tensorflow`: 모델 학습과 예측을 위한 딥러닝 라이브러리.
- `pandas`: 데이터 로딩 및 처리.
- `numpy`: 수학적 연산과 배열 처리.

## 데이터
이 프로젝트에서는 다음과 같은 데이터가 사용되었습니다:
- **gre**: GRE 점수
- **gpa**: 대학 GPA
- **rank**: 학생이 졸업한 대학의 순위 (1부터 4까지)
- **admit**: 대학 합격 여부 (0 = 불합격, 1 = 합격)

## 모델 아키텍처
모델은 다음과 같은 **다층 퍼셉트론(MLP)** 구조로 구성되었습니다:
- **첫 번째 Dense 레이어**: 64개의 유닛, 활성화 함수는 `tanh`
- **두 번째 Dense 레이어**: 128개의 유닛, 활성화 함수는 `tanh`
- **출력 Dense 레이어**: 1개의 유닛, 활성화 함수는 `sigmoid` (이진 분류)

### 모델 컴파일
모델은 `adam` 옵티마이저를 사용하고, 손실 함수는 `binary_crossentropy`를 사용하여 이진 분류 문제를 해결합니다.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
