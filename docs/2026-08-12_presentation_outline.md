# 발표 자료 초안 — STEP6.5 ~ STEP12 (중간발표 이후 진행분)

중간발표에서 STEP6(첫 Trainer 학습)까지 다뤘다는 전제로, 그 이후(STEP6.5 leakage 수정)부터
오늘(STEP12 Grad-CAM)까지 진행한 내용을 슬라이드 단위로 정리했다. 슬라이드 제목 그대로
PPT 슬라이드 제목으로 쓰고, 불릿을 그대로 본문에 옮기면 되도록 구성했다. 숫자는 전부
`docs/2026-08-11_step8_hpo_report.md`, `docs/2026-08-12_step8_11_12_final_report.md`와
실제 결과 CSV에서 재확인한 값이다.

---

### Slide 1. 개요 — 오늘까지의 진행 범위

- 중간발표: SMILES → 이미지 렌더링 → CNN 멀티태스크 회귀(Tg/Density/FFV/Tc/Rg) 파이프라인 구축, 첫 Trainer 학습(STEP6)까지
- 이번 발표 범위: **STEP6.5(교차검증 검증) → STEP7(backbone) → STEP9(loss) → STEP8(HPO) → STEP11(앙상블) → STEP12(해석)**
- 한 줄 요약: "검증 방법을 먼저 바로잡고 → 구조/손실함수/하이퍼파라미터를 단계적으로 개선 → 앙상블로 최종 성능 확보 → Grad-CAM으로 근거 확인"

제안 시각자료: 전체 파이프라인 플로우 다이어그램 (전처리 → 모델 → 검증 → 튜닝 → 앙상블 → 해석)

---

### Slide 2. STEP6.5 — 교차검증 Data Leakage 수정

- 문제: 원래 K-Fold가 sklearn plain `KFold` — 같은 polymer(SMILES)가 train/valid에 동시에 들어갈 수 있는 구조
- 검증: RDKit `Chem.CanonSmiles()`로 표기가 달라도 같은 분자인 경우까지 재검사 → 실제 중복 분자 2쌍 발견
- 과거 5-fold 결과를 동일 조건(seed=42)으로 재현 → **5개 fold 중 3개에서 실제 leakage 발생 확인**
- 조치: `GroupKFold(canonical_smiles)` + leakage assert로 교체, 이후 모든 검증 코드(HPO 포함)에 동일 기준 통일

제안 시각자료: "Fold 1~5 중 leak 발생 여부" 표 (O/X)

---

### Slide 3. STEP7 — Backbone 비교 (ResNet18 vs EfficientNet-B0)

- 동일 조건(5-fold GroupKFold)에서 두 backbone 비교

| Backbone | MAE | RMSE | R² |
|---|---|---|---|
| ResNet18 | 0.3380 | 0.5478 | 0.6874 |
| EfficientNet-B0 | 0.3423 | 0.5512 | 0.6854 |

- ResNet18이 근소 우위(차이는 fold 간 표준편차보다 작아 통계적으로 유의미하진 않음) → **ResNet18을 baseline backbone으로 채택**

제안 시각자료: 두 backbone MAE/RMSE/R² 막대그래프

---

### Slide 4. STEP9 — Loss Architecture 설계

- `losses/` 패키지 신규: `ElementLoss`(MSE/MAE/Huber) × `WeightingStrategy`(Uniform/Static)를 조합 가능한 구조로 분리
- Trainer는 `LossFactory.build(config)`로 loss를 생성 — config만 바꾸면 새 loss 조합을 바로 실험 가능
- 목적: STEP9.4의 4종 비교 실험을 위한 인프라

제안 시각자료: `element_loss × weighting` 조합 구조 다이어그램 (2x2 매트릭스 느낌)

---

### Slide 5. STEP9.4 — Loss 함수 비교 실험

- 4개 실험(ResNet18 고정): Masked MSE, Masked MAE, Masked Huber, Weighted MSE(라벨 개수 역수 가중)

| 실험 | MAE | RMSE | R² |
|---|---|---|---|
| Masked MSE | 0.3295 | 0.5365 | 0.7002 |
| Masked MAE | 0.3245 | 0.5399 | 0.6975 |
| **Masked Huber** | 0.3275 | **0.5357** | **0.7039** |
| Weighted MSE | 0.3397 | 0.5433 | 0.6938 |

- Huber가 RMSE·R² 1위, MAE도 최고(MAE 실험)와 거의 동률 → **Huber loss 채택**
- 라벨 개수 역가중(exp4)은 오히려 4종 중 최하위 — "라벨 적은 property에 가중치를 더 준다"는 직관이 항상 통하지는 않음

제안 시각자료: 4개 실험 MAE/RMSE/R² 막대그래프 (Huber 강조)

---

### Slide 6. STEP8 — 하이퍼파라미터 탐색 (Optuna HPO)

- 절차: **Round1(넓은 범위, 18 trials)** → **Round2(좁힌 범위, 15 trials)** → **Confirmation(5-fold 재실행)**
- 탐색 비용 절감을 위해 fold 1개로만 탐색하고, 마지막에 최종 후보만 5-fold 전체 검증
- Round1에서 dropout=0.2·batch_size=64 조합이 상위권을 독점 → Round2에서 이 조합으로 고정하고 learning_rate/weight_decay만 좁혀서 재탐색
- **최종 채택**: `learning_rate=2.37e-4, weight_decay=7.97e-5, dropout=0.2, batch_size=64`
- Confirmation 5-fold: **MAE 0.317, RMSE 0.521, R² 0.719**

제안 시각자료: Round1 대비 Round2 val_loss 분포(박스플롯) — 대부분 trial이 낮고 조밀한 구간에 모임

---

### Slide 7. STEP8 재현성 검증 (8/11 vs 8/12 재실행)

- 같은 하이퍼파라미터·같은 fold 분할로 다음날 재실행 → **5-fold 평균은 안정적**(MAE/RMSE/R² 차이 1.3% 이내)
- 그러나 **개별 fold 단위는 변동이 큼** — fold 5는 R² 0.781 → 0.629로 0.15 이상 하락
- 데이터 분할은 동일하므로 이 변동은 순수하게 **가중치 초기화·배치 셔플 등 학습 확률성(seed noise)**
- 결론: 단일 fold 숫자는 신뢰하기 어렵고, **5-fold 평균이 대표값**

제안 시각자료: fold별 8/11 vs 8/12 MAE 비교 막대그래프 (변동폭 강조)

---

### Slide 8. STEP11 — 5-fold 앙상블 / OOF 평가

- Confirmation 5개 fold 모델을 그대로 앙상블 멤버로 사용
- **OOF(Out-of-Fold) 평가**: 각 샘플을 "그 샘플이 valid였던 fold의 모델"로만 예측 → leakage 없는 평가

| Property | MAE | RMSE | R² | 라벨 수 |
|---|---|---|---|---|
| Tg | 53.89 | 71.04 | 0.590 | 557 |
| Density | 0.0327 | 0.0620 | **0.820** | 613 |
| FFV | 0.00647 | 0.0133 | 0.791 | 7,892 |
| Tc | 0.0306 | 0.0652 | **0.585** | 867 |
| Rg | 1.716 | 2.510 | 0.703 | 614 |

- 흥미로운 관찰: **라벨 개수와 R²가 단순 비례하지 않음** — Density는 라벨이 적어도(613개) 성능 1위, Tc는 라벨이 더 많아도(867개) 성능 꼴찌
- 해석: property 자체가 2D 골격 이미지에서 얼마나 잘 드러나는가(부피/충전 vs 사슬 간 상호작용)가 라벨 개수보다 중요해 보임

제안 시각자료: property별 R² 막대그래프 (라벨 수를 보조축으로)

---

### Slide 9. STEP11 — Test Set 앙상블 제출

- 5개 fold 모델의 예측을 평균해 최종 제출 파일 생성 (public test set 3개 샘플)
- NaN 없음, 물리적으로 타당한 범위(Tg 66~180, Density 1.05~1.07 등) 확인

제안 시각자료: `ensemble_submission.csv` 표를 그대로 슬라이드에 삽입 (3행이라 작음)

---

### Slide 10. STEP12 — Grad-CAM으로 모델 판단 근거 확인

- ResNet18 layer4를 대상으로 표준 Grad-CAM 적용, property별 3샘플씩 총 15장
- **약 절반(8/15)**은 화학적으로 설명 가능한 위치에 반응:
  - FFV: 입체 장애 큰 다환/스피로 골격 (가장 설득력 있음)
  - Tg: 강직한 방향족-아미드 코어, 비시클릭 고리
  - Tc: fluorene계 융합 방향족 코어
- **나머지는 배경/모서리 아티팩트** — 실패 케이스도 정직하게 공개 (분자가 가늘고 길게 그려진 샘플에서 두드러짐)

제안 시각자료: `results/step12_gradcam/gradcam_FFV.png`(sample 2), `gradcam_Tg.png`(sample 18), `gradcam_Tc.png`(sample 4) 중 2~3장 삽입

---

### Slide 11. 전체 성능 개선 추이

| 단계 | 설정 | MAE | RMSE | R² |
|---|---|---|---|---|
| STEP7 baseline | ResNet18, MSE | 0.3380 | 0.5478 | 0.6874 |
| STEP9.4 loss 개선 | ResNet18, Huber | 0.3275 | 0.5357 | 0.7039 |
| STEP8 HPO 최종 | ResNet18, Huber, 튜닝 완료 | 0.3171 | 0.5209 | 0.7192 |

- backbone 선택 → loss 개선 → HPO 튜닝을 거치며 MAE/RMSE는 꾸준히 감소, R²는 꾸준히 상승
- (val_loss는 loss 함수 스케일이 달라 단계 간 비교에서 제외 — MAE/RMSE/R²만 공정 비교 가능)

제안 시각자료: 3단계 개선 추이를 보여주는 라인 차트 (MAE 감소, R² 상승 2개 축)

---

### Slide 12. 한계점 및 다음 단계

**한계점**
- fold 단위 재현성 noise (Slide 7) — 개별 fold 숫자만으로 결론 내리지 않도록 주의
- OOF "overall" 지표는 5개 property를 단순 평균한 값이라 스케일이 섞여 의미 없음 → property별 표만 사용
- Grad-CAM 15장 중 약 40%는 배경/모서리 아티팩트 — 정성적 참고용이지 정량적 검증은 아님

**다음 단계**
- (필요 시) 앙상블 다양성 확대: EfficientNet-B0, MAE-loss 모델 등 다른 설정의 멤버 추가 검토
- 학회 short paper 작성 — Method/Results/Discussion/Limitations 섹션 배치 초안은
  `docs/2026-08-12_step8_11_12_final_report.md` 5절에 이미 정리돼 있음

---

## 부록 — 슬라이드별 원본 근거 파일

| 슬라이드 | 근거 파일 |
|---|---|
| 2 (leakage) | `docs/2026-08-10_status_review.md`, `training/training real/cross_validation.py` |
| 3 (backbone) | `results/kfold_results_resnet18.csv`, `results/kfold_results_efficientnet_b0.csv` |
| 4~5 (loss) | `results/step9_4/loss_comparison.csv`, `losses/factory.py` |
| 6~7 (HPO) | `docs/2026-08-11_step8_hpo_report.md`, `results/step8_hpo/round{1,2}/trials.csv`, `results/step8_hpo/kfold_results_confirmation.csv` |
| 8~9 (앙상블) | `results/step11_ensemble/oof_per_property.csv`, `ensemble_submission.csv` |
| 10 (Grad-CAM) | `results/step12_gradcam/gradcam_*.png` |
| 11 (전체 추이) | `docs/2026-08-12_step8_11_12_final_report.md` 4절 |
