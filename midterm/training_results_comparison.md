# 학습 결과 비교 보고서

작성일: 2026-07-26  
범위: `dry_run_*` 산출물을 제외한 실제 학습 결과만 비교

## 1. 비교 대상과 정리 기준

동일 파일의 복사본은 하나의 실험으로 묶었다. 따라서 다음 네 결과를 확인했다.

| 구분 | 대표 산출물 | 데이터 | 검증 방식 | 상태 |
|---|---|---:|---|---|
| Baseline (single split) | `results/baseline` | train 7,973개 / test 3개 | 고정 10% validation split | 비교 가능 |
| Merged (single split) | `results/merged` | train 8,974개 / test 3개 | 고정 10% validation split | 비교 가능 |
| Merged before K-fold | `results/result/merged_before_kfold` | merged 데이터 | 단일 분할 추정 | 예측 비교만 가능 |
| Merged 5-fold | `results/result/merged_after_kfold` | merged train 8,974개 | 5-fold CV | 주요 일반화 성능 |

제외한 항목은 이름이 `dry_run_*`인 metric summary, prediction, learning curve, report 전부이다. 또한 아래는 동일 내용의 중복 복사본으로 별도 실험으로 세지 않았다.

- `results/result/baseline` = `results/baseline`
- `results/result/merged` = `results/merged`
- `reports/report` = `results/result/merged_before_kfold`
- `results/result/kfold_results.csv` = `results/result/merged_after_kfold/kfold_results.csv`

## 2. 실험 설정

모든 실제 학습은 ImageNet pretrained ResNet-18, dropout 0.3, MSE loss, cosine scheduler, batch size 32, learning rate 1e-4, 최대 30 epoch, seed 42를 사용했다. 데이터 병합은 학습 샘플을 7,973개에서 8,974개로 1,001개(+12.6%) 늘렸다.

## 3. 단일 분할 학습 비교

| 항목 | Baseline | Merged | 해석 |
|---|---:|---:|---|
| best validation loss | **0.131714** | 0.133741 | baseline이 0.002027(약 1.54%) 낮음 |
| best epoch (0-indexed) | 27 | 29 | 둘 다 거의 최대 epoch까지 학습 |
| 실행 epoch | 30 | 30 | 동일 |
| 테스트 예측 행 수 | 3 | 3 | 테스트 정답이 없어 실제 오차 비교 불가 |

단일 분할의 validation loss만 보면 baseline이 근소하게 우세하다. 다만 두 데이터셋의 validation 구성과 정규화 통계가 달라, 이 수치만으로 병합 데이터가 성능을 악화시켰다고 단정할 수는 없다. 특히 merged는 더 많은 학습 데이터를 사용했고 최종 모델도 30 epoch까지 안정적으로 학습됐다.

`merged_before_kfold`도 3개 테스트 행에 대한 예측과 learning curve는 남아 있으나, 해당 체크포인트는 손상되어 재로딩되지 않았다. 따라서 이 결과는 정량적 성능 비교표에서는 제외하고 예측 변화 확인용으로만 취급했다.

## 4. Merged 5-fold 교차검증 결과

5-fold는 각 fold의 최저 validation-loss epoch 기준 결과이며, fold별 train 데이터만으로 label 통계를 계산하도록 구현되어 있다.

| 지표 | 평균 | 표준편차 | 최저~최고 |
|---|---:|---:|---:|
| Train loss | 0.132714 | 0.053487 | 0.100023 ~ 0.227938 |
| Validation loss | 0.252765 | 0.068145 | 0.144034 ~ 0.333088 |
| MAE | 0.337469 | 0.027584 | 0.305701 ~ 0.380673 |
| RMSE | 0.541040 | 0.056936 | 0.451812 ~ 0.606622 |
| R² | 0.698764 | 0.050315 | 0.656607 ~ 0.778561 |

Fold 4가 가장 좋았고(R² 0.778561, RMSE 0.451812), Fold 2가 가장 낮은 R²(0.656607)를 보였다. R²의 표준편차가 약 0.05이므로 split에 따라 성능 변동은 있지만, 모든 fold에서 R²가 0.65 이상으로 양수여 merged 모델이 평균적으로 의미 있는 설명력을 보였다.

## 5. 결론 및 권고

1. 제출 또는 후속 모델 선택의 기준은 단일 10% split보다 **merged 5-fold CV** 결과(MAE 0.3375, RMSE 0.5410, R² 0.6988)를 우선하는 것이 타당하다.
2. Baseline과 merged 단일 분할의 손실 차이는 1.54%로 작고 평가 분할이 다르므로, 병합 효과를 엄밀히 비교하려면 두 데이터셋에 같은 fold/동일 지표를 적용한 추가 CV가 필요하다.
3. `merged_before_kfold` 및 `merged_after_kfold`의 일부 최상위 체크포인트가 손상되어 재로딩되지 않는다. 결과 CSV와 prediction은 보존하되, 재현성을 위해 정상 체크포인트를 다시 생성하거나 파일 무결성을 확인해야 한다.
4. 테스트 세트는 3행이며 정답이 없으므로 prediction 평균·분산은 모델 성능 지표가 아니다. 외부 평가 점수가 확보되기 전에는 이를 모델 우열 근거로 사용하지 않는다.

## 근거 파일

- `results/baseline/checkpoints/best_model.pth`
- `results/merged/checkpoints/best_model.pth`
- `results/result/merged_after_kfold/kfold_results.csv`
- `data/processed_baseline/processed_metadata.csv`
- `data/processed_merged/processed_metadata.csv`
