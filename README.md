
# 📖 Overview
<!-- 재활용 품목 분류를 위한 Object Detection -->
본 프로젝트는 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나인 OCR (Optical Character Recognition) 기술에 사용된 모델의 성능 개선을 목표로 한다. 데이터 분석을 통해 평가 데이터에 적용된 노이즈를 발견하였다. 이를 해결하고자 데이터 증강, 초해상도, 디노이징, 배경 제거 및 앙상블 등의 전략 수립 후 실험을 수행하였다. 이들 중 가장 유의미한 결과를 얻은 초해상도, 앙상블 전략을 적용하여 2위의 성적을 거두었다.


## 🏆 Rank

<center>
<img src="https://github.com/FinalCold/Programmers/assets/67350632/b542e661-6b18-4bb1-a0d7-9f9960d753c6" width="700" height="">
<div align="center">
  <sup>Test dataset(Public)
</sup>
</div>
</center>

<center>
<img src="https://github.com/FinalCold/Programmers/assets/67350632/95e21271-6941-4420-a4e2-e9d86d57e937" width="700" height="">
<div align="center">
  <sup>Test dataset(Private)
</sup>
</div>
</center>

## 🪄 How to run
- 학습 시 필요한 정보를 tools 폴더 내 코드를 활용해서 prepocessing 후 코드 실행
```python
python ./code/train.py --train_dataset_dir /path/to/dataset --model_dir ...
python ./code/inference.py --model_dir /path/to/model --data_dir /path/to/data
```

## 🗂 Dataset

<img width="700" alt="1" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/11856a07-3b82-412a-9d22-dda19d01cdf4">

<img width="1495" alt="2" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/3fff6fec-3813-406b-ae28-16af03d8d844">

- **Images & Size :** Train: 100, Test: 100, (600~2500) * (600~3500)

## 📃 Metric
- OCR 은 글자 검출, 글자 인식, 정렬기 등의 모듈로 이루어져 있음. 본 대회에서는 글자 검출 task 만을 해결함. 검출 성능 평가를 위한 지표로 Precision, Recall, F1 Score를 사용함.

<img width="800" alt="0" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/458c68ac-10be-4b85-b8a2-a4808eb21129">

# Team CV-01

## 👬🏼 Members 
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/minyun-e"><img height="110px"  src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/6ac5b0db-2f18-4e80-a571-77c0812c0bdc"></a>
            <br/>
            <a href="https://github.com/minyun-e"><strong>김민윤</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/2018007956"><img height="110px"  src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/cabba669-dda2-4ead-9f73-00128c0ae175"/></a>
            <br/>
            <a href="https://github.com/2018007956"><strong>김채아</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Eddie-JUB"><img height="110px"  src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/2829c82d-ecc8-49fd-9cb3-ae642fbe7513"/></a>
            <br/>
            <a href="https://github.com/Eddie-JUB"><strong>배종욱</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/FinalCold"><img height="110px" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/fdeb0582-a6f1-4d70-9d08-dc2f9639d7a5"/></a>
            <br />
            <a href="https://github.com/FinalCold"><strong>박찬종</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/MalMyeong"><img height="110px" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/0583f648-d097-44d9-9f05-58102434f42d"/></a>
            <br />
            <a href="https://github.com/MalMyeong"><strong>조명현</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/classaen7"><img height="110px"  src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/2806abc1-5913-4906-b44b-d8b92d7c5aa5"/></a>
              <br />
              <a href="https://github.com/classaen7"><strong>최시현</strong></a>
              <br />
          </td>
    </tr>
</table>  
      
                

## 👩‍💻 Roles

<table>
  <tr>
    <th>Name</th>
    <th>Common</th>
    <th>Role</th>
  </tr>
  <tr>
    <td>김민윤</td>
    <td rowspan="6">EDA,<br> 모델 학습 결과 분석</td>
    <td>Image Background Remove</td>
  </tr>
  <tr>
    <td>김채아</td>
    <td>WandB 세팅, 학습 속도 최적화 실험</td>
  </tr>
  <tr>
    <td>배종욱</td>
    <td>Data Augmentation, Super Resolution, Denoise</td>
  </tr>
  <tr>
    <td>박찬종</td>
    <td>Pickle Data Generation, Data Augmentation, Corner Crop</td>
  </tr>
  <tr>
    <td>조명현</td>
    <td>Denoise, Text sharpening</td>
  </tr>
  <tr>
    <td>최시현</td>
    <td>Annotation 시각화, Box Filtering & Ensemble</td>
  </tr>
</table>


</br>

## 💻 Enviroments

- Language: Python 3.10.13
- Hardwares: Intel(R) Xeon(R) Gold 5120, Tesla V100-SXM2 32GB × 6
- Framework: Pytorch, Numpy
- Cowork Tools: Github, Weight and Bias, Notion, Discord, Zoom, Google calendar

</br>

# 📊 Project
## 🔎 EDA

<img width="1495" alt="2" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/3fff6fec-3813-406b-ae28-16af03d8d844">

> ### Train Dataset
- 학습 데이터셋 분석에서 세로 이미지, 가로 이미지, 구겨진 이미지 총 세가지 특성을 파악함.
- 또한, 노이즈가 거의 없는 클린한 데이터임을 파악함.

> ### Test Dataset
- 평가 데이터셋 분석 과정에서 학습 데이터셋과는 대조적으로 이미지 내에 다양한 노이즈가 존재하는 것을 확인함.

> ### Limitation in UFO Format
- UFO 포맷은 Albumentation과 같은 라이브러리에서 포맷을 지원하지 않아 Vertice를 변경하는 Transformation에 대해서 Numpy를 이용하며, 이는 학습 시 CPU 연산을 이유로 병목현상이 발생하여 학습 시간이 오래걸린다는 단점을 확인함.

<img width="861" alt="3" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/4b1428a9-bdf6-4081-83bf-04a3a4fb5ea5">


## 🔬 Methods


> ### Pickle Data Generation

- 학습 데이터를 Pickle 파일로 변환
```python
python ./code/to_pickle.py
```

<img width="969" alt="4" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/957ed0ae-c51f-498b-a1d7-1af22ce851e2">

- Pickle 파일로 학습 데이터를 생성하여 학습 시간을 기존 1에폭 당 15분 걸리던 학습 시간을 10초 이내로 단축함.

> ### Augmentation
- 평가 데이터셋과 유사한 노이즈라고 생각되는 다양한 Augmentation 기법을 적용하여 실험 진행함.

<img width="1238" alt="5" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/f2c96b75-9bb3-4003-ab07-3959052372ab">

| Augmentation | F1 score | Recall | Precision |
| :---: | :---: | :---: | :---: |
| Default_CJ, Default_N | 0.9119 | 0.8647 | 0.9645 |
| Default_CJ, Default_N, S&P | 0.9024 | 0.8493 | 0.9627 |
| Default_CJ, Default_N, GB, B | 0.8967 | 0.849 | 0.9501 |

| Augmentation | F1 score | Recall | Precision |
| :---: | :---: | :---: | :---: |
| Default_CJ, Default_N | 0.8779 | 0.9214 | 0.8383 |
| CJ, N | 0.8532 | 0.8692 | 0.8377 |
| GN, N | 0.7978 | 0.7755 | 0.8214 |
| CJ, GB, B, N | 0.8843 | 0.8735 | 0.8954 |
| CJ, GB, B, GN, N | 0.8634 | 0.8618 | 0.8651 |
| CJ, GB, B, HSV, N | 0.8807 | 0.9019 | 0.8604 |
| CJ, GB, HSV, N | 0.9124 | 0.8665 | 0.9651 |

> ### Denoise
<img width="975" alt="6" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/6f88cc5b-a5b2-4e49-aa96-348c3f0979a3">

</br>

- 평가 데이터셋 내의 노이즈는 모델의 성능에 부정적인 영향을 미칠 가능성이 높다고 판단함.
- Cycle GAN을 활용하여 이미지 내의 상당수의 노이즈를 제거함.

</br>

> ### Background Remove

<img width="1141" alt="7" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/ec03a7ba-700c-46b1-a9f4-d32c9cb700cd">

<img width="1140" alt="8" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/861f0706-3e34-4e5b-ad34-89a149a693e9">

</br>

- 평가 데이터셋 추론 결과 문서 외부 배경 부분에서 다수의 노이즈 검출 확인함.
- 모델의 배경 노이즈 검출이 줄어들 것을 가정하여 배경 제거 후 평가 데이터셋에 대한 추론 수행함.

| Dataset | F1 score | Recall | Precision |
| :---: | :---: | :---: | :---: |
| Original | 0.9106 | 0.915 | 0.9061 |
| Background removed | 0.8608 | 0.8758 | 0.8463 |


    

</br>

> ### Super Resolution

<img width="931" alt="9" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/a149e783-5c51-43f2-912d-ae5a67959d01">

<img width="1280" alt="10" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/f63e22d5-585e-499e-b917-5d5eccc43055">

</br>

- 제공된 데이터셋의 경우 글자들의 크기가 대부분은 작은 것을 확인하여 초해상도 기법을 적용하여 원본 이미지 대비 2배, 4배 해상도의 이미지를 생성함.
- 서버 GPU 메모리 한계로 원본 이미지에 직접 SR을 적용하기 어려워서 이미지를 8등분으로 자른 뒤 SR을 적용함.

| Dataset | F1 score | Recall | Precision |
| :---: | :---: | :---: | :---: |
| Original | 0.9106 | 0.915 | 0.9061 |
| SR x2 | 0.9381 | 0.9392 | 0.9369 |
| SR x4 | 0.941 | 0.9369 | 0.9451 |

</br>


> ### Corner Crop

<img width="555" alt="12" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/88a77da2-4239-4a66-88d5-ddd2d2649097">

- 이미지의 우측 상단에 QR code 옆 세로 글씨에 대한 검출 성능이 부족하여 의도적으로 Coner 부분을 포함시켜 학습을 진행함

</br>

> ### Ensemble

<img width="1181" alt="13" src="https://github.com/Eddie-JUB/Portfolio/assets/71426994/75439ff9-50b4-4205-8716-936c51b6cf86">

- 문자 검출 시에 Confidence Score를 산출하지 않기 때문에 노이즈라고 간주되는 낮은 신뢰도 영역을 식별하는데 어려움이 있음.
- Noise Filtering : 설정한 임계값보다 적은 수 의 모델이 예측한 영역의 경우 노이즈라고 간주
- Small box Filtering : 작은 박스들 중 IoU 값이 낮아 앙상블을 통해 합쳐지지 않는 경우 제거

| Ensemble | F1 score | Recall | Precision |
| :---: | :---: | :---: | :---: |
| Model Average | 0.9364 | 0.9357 | 0.9371 |
| Noise Filtering | 0.9485 | 0.9479 | 0.9491 |
| Noise Filtering & Small box filtering | 0.9503 | 0.9479 | 0.9526 |



</br>

# 📈 Experimental Result

| No | Ensemble | Dataset | Image Size | F1 score | Recall | Precision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Single | Original | $[1024,1536,2048]$ | 0.9106 | 0.915 | 0.9061 |
| 1 | Single | Original | $[1024,1536,2048,4096]$ | 0.9303 | 0.9312 | 0.9294 |
| 2 | Single | SR x2 | $[1024,1536,2048,4096]$ | 0.9381 | 0.9392 | 0.9369 |
| 3 | Single | SR x4 | $[1024,1536,2048,4096,8192]$ | 0.9410 | 0.9369 | 0.9451 |
| 4 | Single | SR x4 | $[1024,1536,2048,4096$, <br> $8192,12288,16384]$ | 0.9389 | 0.9355 | 0.9424 |
| $\mathbf{5}$ | $\mathbf{1}+\mathbf{2}+\mathbf{3}$ | - | - | $\mathbf{0 . 9 5 0 6}$ | $\mathbf{0 . 9 5 0 3}$ | $\mathbf{0 . 9 5 0 8}$ |

</br>

# Conclusion & Discussion

## Conclusion
본 대회의 목표는 병원 영수증에 포함된 글자를 정확히 검출하는 것이 목표이다. EDA를 통해 학습 데이터셋과 달리 평가 데이터셋에 많은 노이즈가 포함된 것을 확인하였다. 이를 해결하고자 배경 제거, 노이즈 제거, 초해상도 기법, 데이터 증강을 적용하였다. 특히 초해상도 기법을 통해 Recall과 Precision 모두 큰 폭의 성능 향상을 확인할 수 있었다. 이후 결과 데이터 분석을 통해 앙상블을 적용하여 좋은 성적을 낼 수 있었다. 

## Discussion
> ###  데이터 클렌징 & 리레이블링
지난 대회에서 훈련 데이터셋과 평가 데이터셋의 레이블링에 대한 비슷한 양상의 오류가 존재하였다. 본 대회도 비슷한 양상으로 판단하여 클렌징 작업과 리레이블링 작업을 생략하였다.

> ### 대회 가이드라인 변경
대회 진행 중 평가 데이터셋 전처리 가이드라인 변경으로 평가 데이터셋에 대한 다양한 전처리 기법을 최대한 활용하지 못했다.

> ### 평가 데이터와 유사한 검증 데이터셋 구축
학습 데이터셋과 달리 많은 노이즈가 포함된 평가 데이터셋이 주어졌다. 따라서 평가 데이터셋과 경향성이 유사한 검증 데이터셋을 찾기 어려웠다.

> ###  K-fold
bbox별 confidence score가 존재하지 않아 모델의 voting이 이루어질 수 없다. 따라서 일반적인 앙상블 방식과는 다르게 필터링 조건들을 직접 구현하였다.

