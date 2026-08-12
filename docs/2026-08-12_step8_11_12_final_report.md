# 2026-08-12 최종 결과 리포트 — STEP8 Confirmation 재실행 / STEP11 앙상블 / STEP12 Grad-CAM

오늘 진행한 세 작업(STEP8 Confirmation 재실행, STEP11 앙상블, STEP12 Grad-CAM)의
결과를 실제 결과 파일 기준으로 정리한다. 숫자는 모두 아래 파일에서 직접 읽었다.

- `results/step8_hpo/kfold_results_confirmation.csv` (오늘 재실행분)
- `results/step11_ensemble/oof_overall.csv`, `oof_per_property.csv`, `ensemble_submission.csv`
- `results/step12_gradcam/gradcam_{Tg,Density,FFV,Tc,Rg}.png`
- (비교용) `results/kfold_results_resnet18.csv`, `results/step9_4/loss_comparison.csv`,
  `results/step8_hpo/round2/best_params.json`, `docs/2026-08-11_step8_hpo_report.md`

---

## 1. STEP8 Confirmation 재실행 — 재현성 검증

### 배경

어제(8/11) GPU 서버가 인터넷 연결 문제로 중간에 끊겨, `round2/best_params.json`
(`lr=2.369e-4, wd=7.969e-5, dropout=0.2, batch_size=64`, resnet18 + huber/uniform)의
5-fold Confirmation을 오늘 처음부터 재실행했다. config와 hyperparameter는 어제와
동일하고, fold 분할도 `canonical_smiles` 기준 GroupKFold(결정적, 셔플 랜덤성 없음)라
**데이터 분할 자체는 두 실행에서 완전히 동일**하다. 달라질 수 있는 건 가중치 초기화·
배치 셔플 순서 등 학습 과정의 확률성뿐이다.

### Fold별 비교 (8/11 최초 실행 vs 8/12 재실행)

| Fold | 8/11 MAE | 8/12 MAE | Δ | 8/11 R² | 8/12 R² | Δ |
|---|---|---|---|---|---|---|
| 1 | 0.32744 | 0.31919 | -0.00825 | 0.69902 | 0.72765 | +0.02863 |
| 2 | 0.31083 | 0.27424 | -0.03659 | 0.74616 | 0.77341 | +0.02725 |
| 3 | 0.30996 | 0.35478 | +0.04482 | 0.71069 | 0.69436 | -0.01633 |
| 4 | 0.34208 | 0.30570 | -0.03638 | 0.65885 | 0.72365 | +0.06480 |
| 5 | 0.29527 | 0.35293 | +0.05766 | 0.78137 | 0.62865 | -0.15272 |

### 평균 ± 표준편차 비교

| 지표 | 8/11 (최초) | 8/12 (재실행) | 상대 차이 |
|---|---|---|---|
| MAE | 0.31712 ± 0.01801 | 0.32137 ± 0.03388 | +1.3% |
| RMSE | 0.52091 ± 0.07853 | 0.52791 ± 0.09052 | +1.3% |
| R² | 0.71922 ± 0.04667 | 0.70954 ± 0.05334 | -1.3% |

**해석**

- **5-fold 평균(집계 지표) 기준으로는 재현성이 양호하다.** MAE/RMSE/R² 모두 1.3%
  수준의 차이만 나며, 이는 8/11 실행 자체의 fold 간 표준편차(MAE std 0.018)보다도
  작은 변동폭이다.
- 반면 **개별 fold 단위로 보면 변동이 상당히 크다.** 특히 fold 5는 R² 0.781 →
  0.629로 0.15 이상 떨어졌고, fold 4는 반대로 0.659 → 0.724로 0.06 이상 올랐다.
  데이터 분할이 동일하므로 이 변동은 순수하게 **학습 과정의 확률성(가중치 초기화,
  배치 셔플)에서 오는 seed noise**로 봐야 한다.
- 이는 어제 STEP8 HPO 리포트에서 관찰된 패턴(Round1 trial 14, Round2 trial 9/11처럼
  하이퍼파라미터가 거의 같은데도 단일 fold 평가에서 val_loss가 크게 튀는 현상)과
  같은 맥락이다 — **단일 fold 결과는 신뢰하기 어렵고, 5-fold 평균으로 봐야 재현
  가능한 숫자가 나온다**는 점이 오늘 재실행으로 다시 한번 확인됐다.
- 결론: **"MAE 0.317~0.321, R² 0.71~0.72" 수준을 STEP8 confirmation의 최종
  성능으로 보고하는 것이 타당**하다 (개별 fold 표는 참고용, 평균이 대표값).

---

## 2. STEP11 앙상블

### 2.1 OOF(Out-of-Fold) 평가

`ensemble/oof_evaluation.py`는 각 샘플을 **그 샘플이 validation이었던 fold의
모델로만** 예측한다 (leakage 없음). 대상은 train pool 전체이며, property마다
라벨이 있는 샘플만 집계한다.

**Property별 성능 (원본 물리 단위, `oof_per_property.csv`)**

| Property | MAE | RMSE | R² | n (라벨 수) |
|---|---|---|---|---|
| Tg | 53.8862 | 71.0433 | 0.5902 | 557 |
| Density | 0.03266 | 0.06198 | 0.8199 | 613 |
| FFV | 0.006468 | 0.013290 | 0.7909 | 7,892 |
| Tc | 0.03055 | 0.06524 | 0.5846 | 867 |
| Rg | 1.71636 | 2.51023 | 0.7028 | 614 |

**`oof_overall.csv`의 "overall" 수치(MAE 11.13, RMSE 14.74, R² 0.6977)에 대한 주의**

`ensemble_common.py`의 `masked_metrics()` 코드를 확인한 결과, 이 "overall" 값은
개별 예측 오차를 다시 풀링한 게 아니라 **위 5개 property MAE/RMSE/R²를 그대로
단순 평균한 값**이다(가중치 없음). 문제는 5개 property가 서로 완전히 다른 물리
단위/스케일을 갖는다는 점이다 — Tg는 ~54, Rg는 ~1.7인 반면 FFV/Density/Tc는
0.01~0.03 수준이다. 그 결과 "overall MAE 11.13"은 사실상 **Tg 하나의 스케일에
지배당한 숫자**이고, 실질적인 성능 지표로서 의미가 없다. **논문/리포트에는 반드시
property별 표를 그대로 인용해야 하며, overall 행은 사용하지 않는 것을 권장한다.**

**z-score 스케일이었던 STEP8 confirmation MAE(0.3171/0.3214)와의 관계**

STEP8 confirmation의 MAE(0.317~0.321)는 5개 property를 각 fold의 train 쪽 평균/
표준편차로 **z-score 정규화한 뒤**, 정규화된 스케일에서 masked MAE를 property
축까지 한꺼번에 평균 낸 **단일 무차원(unitless) 지표**다. 반면 STEP11 OOF의
property별 MAE(53.89, 0.033, 0.0065, 0.031, 1.72)는 **각 property의 원본 물리
단위로 역정규화(denormalize)한 뒤** 계산한 값이라 서로 스케일이 다른 5개의 독립된
숫자다. 즉:

- STEP8 MAE 0.317 → "정규화 공간에서 평균적으로 표준편차의 0.32배만큼 틀린다"는
  뜻의 **모델 선택/튜닝용 단일 요약 지표**.
- STEP11 OOF property별 표 → "Tg는 평균 약 54(단위) 틀리고, Density는 약 0.033
  틀린다"는 **실제 배포/해석용 단위 성능**.

둘은 같은 모델을 다른 잣대로 잰 것이라 숫자를 직접 비교(예: "0.317 vs 11.13")하면
안 되고, **정규화 지표는 학습/튜닝 단계의 상대 비교용, property별 원본 단위 지표는
최종 보고용**으로 역할이 다르다는 점을 리포트/논문에 명시하는 게 좋다.

### 2.2 Property별 성능 차이 — 라벨 개수와의 관계

라벨 개수(n) 오름차순: Tg(557) < Density(613) < Rg(614) < Tc(867) < FFV(7,892)
R² 오름차순: Tc(0.585) < Tg(0.590) < Rg(0.703) < FFV(0.791) < Density(0.820)

"라벨이 적을수록 R²가 낮다"는 가설은 **부분적으로만 맞다.** FFV는 라벨이
압도적으로 많고(7,892) R²도 두 번째로 높아 가설과 일치하지만, 그 반례가 뚜렷하다:

- **Density**는 라벨이 613개로 두 번째로 적은데도 **R²가 가장 높다(0.820)**.
- **Tc**는 라벨이 867개로 Density·Rg보다 많은데도 **R²가 가장 낮다(0.585)**.

즉 라벨 개수만으로 성능 순위를 설명할 수 없고, **property 자체의 예측 난이도
(2D 골격 이미지에서 얼마나 잘 드러나는 물성인가)가 더 큰 영향을 주는 것으로
보인다.** Density·FFV는 분자의 부피/충전 방식과 비교적 직접적으로 연결되는 반면,
Tg·Tc는 사슬 간 상호작용·3차원 배열 등 2D 골격 구조 이미지만으로는 포착하기
어려운 요인에 더 좌우될 가능성이 있다 — 이는 STEP9.4 loss 비교에서 "라벨 개수
역수 가중(exp4)이 오히려 최하위였다"는 관찰과도 방향이 맞는다(단순히 라벨 적은
property에 가중치를 더 준다고 성능이 좋아지지 않았음).

### 2.3 Test set 앙상블

`test_ensemble.py`로 5개 fold 모델의 예측을 평균해 `ensemble_submission.csv`를
생성했다. 이 competition의 public test set은 원래 샘플 수가 매우 적어(3개), 결과
파일도 3행뿐이다 — 정상이다.

| id | Tg | Density | FFV | Tc | Rg |
|---|---|---|---|---|---|
| 1109053969 | 151.90 | 1.0745 | 0.3752 | 0.2950 | 21.717 |
| 1422188626 | 180.01 | 1.0634 | 0.3753 | 0.2462 | 20.829 |
| 2032016830 | 66.60 | 1.0524 | 0.3500 | 0.2207 | 14.942 |

코드(`test_ensemble.py`)에 내장된 sanity check 2종 — ① 5개 fold 모델 예측이
서로 다른지(diversity, 체크포인트 로딩 버그 방지용), ② 예측값에 NaN이 없는지 —
는 콘솔에만 출력되고 별도 로그 파일로 저장되지 않아 이번 리포트에서 직접 재확인은
못 했다. 다만 `ensemble_submission.csv`를 직접 열어본 결과 **NaN은 없고**, 3개
샘플의 5개 property 값이 모두 물리적으로 그럴듯한 범위(Tg 66~180, Density
~1.05~1.07, FFV ~0.35~0.38 등)에 있어 눈에 띄는 이상은 없다.

---

## 3. STEP12 Grad-CAM

`gradcam.py`는 앙상블 fold 1 모델을 대표로 사용해, ResNet18 backbone
(`feature_extractor`)의 **layer4**(마지막 conv block)를 target layer로 표준
Grad-CAM을 적용했다. Property마다 라벨이 있는 샘플 3개씩, 총 15장의 히트맵을
`results/step12_gradcam/`에 저장했다.

### Property별 소견 (육안 확인)

**Tg** ([gradcam_Tg.png](../results/step12_gradcam/gradcam_Tg.png))
- sample 18: 두 방향족 고리를 잇는 **아미드(-C(=O)NH-) 연결부**에 히트맵이
  집중됨 — 강직한(rigid) 방향족-아미드 코어는 Tg를 높이는 대표적 구조 요인이라
  화학적으로 타당한 위치다.
- sample 37: **비시클릭(노보난형) 고리** 자체에 집중 — 역시 사슬의 강직성/자유
  회전 억제와 관련된 구조라 합리적이다.
- sample 71: 히트맵이 분자 골격이 아니라 **이미지 우상단 모서리(배경)**에 몰려
  있다 — 화학적으로 의미 있는 위치가 아니라, 사슬이 이미지 프레임 왼쪽에 치우쳐
  그려지면서 생긴 것으로 보이는 경계 아티팩트에 가깝다.

**Density** ([gradcam_Density.png](../results/step12_gradcam/gradcam_Density.png))
- sample 4: 니트릴(-C≡N)로 끝나는 **두 알킬 사슬** 쪽에 집중되고, 오른쪽의 부피
  큰 fluorene계 방향족 코어는 거의 반응이 없음 — density는 보통 부피가 큰
  방향족/충전 구조가 크게 기여하는 편이라, 이 케이스는 다소 의외의 위치다(극성
  니트릴기 자체가 패킹 밀도에 기여했을 가능성도 있으나 확정적이지 않음).
- sample 27: 히트맵이 거의 없음(배경색만 균일) — 이 샘플에 대해서는 모델이
  뚜렷한 근거 영역을 찾지 못한 것으로 보인다.
- sample 32: 니트릴-싸이오에터-에스터로 이어지는 **극성 연결부(heteroatom
  linker)**에 집중 — 극성기 밀집 구간이 패킹/밀도에 영향을 준다는 해석과 부합해
  비교적 타당하다.

**FFV** ([gradcam_FFV.png](../results/step12_gradcam/gradcam_FFV.png))
- sample 0, 1: 부피가 크고 입체 장애가 있는 **다환/스피로 골격 구간** 전반에
  넓게 반응 — 자유 부피는 사슬의 입체적 얽힘/충전 저해에서 나오므로 타당하다.
- sample 2: 여러 아릴기가 밀집한 **육치환 벤젠 중심 코어**에 정확히 집중 —
  입체 장애가 가장 큰 위치를 정확히 짚고 있어 5개 property 중 가장 화학적으로
  설득력 있는 케이스다.

**Tc** ([gradcam_Tc.png](../results/step12_gradcam/gradcam_Tc.png))
- sample 0: (Density sample 32와 유사한 골격) 사슬 말단부에 집중, 골격과 겹치는
  부분은 일부뿐 — 애매함.
- sample 4: **fluorene계 융합 방향족 코어**에 집중 — 강직한 공액 구조가 열전달
  경로와 관련된다는 통념과 부합해 타당하다.
- sample 8: 히트맵이 분자와 전혀 겹치지 않고 **이미지 상단 배경**에 위치 — 명백한
  실패/아티팩트 케이스.

**Rg** ([gradcam_Rg.png](../results/step12_gradcam/gradcam_Rg.png))
- sample 4: **부피가 큰 융합 고리 코어**에 집중 — 사슬의 전체적 부피/분지 정도가
  회전반경에 영향을 준다는 점에서 타당하다.
- sample 27: 히트맵 없음(반응 없음).
- sample 32: 히트맵이 분자 왼쪽 하단의 **빈 배경**에 위치 — 골격과 겹치지 않는
  아티팩트.

### 종합 소견

15장 중 **약 절반(Tg 2/3, FFV 3/3, Density 1/3, Tc 1/3, Rg 1/3 ≈ 8/15)**은
방향족/강직 코어, 극성 연결부, 입체 장애가 큰 치환 구조 등 **화학적으로 설명
가능한 영역**에 히트맵이 집중돼 있다. 반면 나머지(Tg 1장, Density 2장, Tc 1장,
Rg 2장 = 6~7장)는 **분자 골격과 겹치지 않는 배경/모서리에 히트맵이 몰리거나
반응이 거의 없는 실패 케이스**다. 특히 이런 실패는 사슬이 가늘고 길게 그려져
이미지 프레임의 상당 부분이 빈 배경으로 채워지는 샘플에서 두드러지는 경향이
보인다 — layer4 feature map의 낮은 공간 해상도(ResNet18 기준 7×7 수준)와 결합돼
경계 영역에 가짜 신호가 생기는 것으로 추정된다. 결론적으로 **Grad-CAM 결과가
모델의 판단 근거를 어느 정도 뒷받침하지만, 정성적 해석 목적의 예시일 뿐 정량적
신뢰도 검증은 아니라는 점을 논문에 명시하는 게 안전**하다.

---

## 4. STEP7 → STEP12 전체 진행 요약

| STEP | 내용 | 방법 | 핵심 결과 |
|---|---|---|---|
| STEP7 | Backbone 비교 | ResNet18 vs EfficientNet-B0, 5-fold GroupKFold(canonical_smiles) | ResNet18 MAE 0.3380 vs EfficientNet-B0 0.3423 — ResNet18 근소 우위(통계적 유의성은 낮음), ResNet18 채택 |
| STEP9 | Loss Architecture 구현 | `losses/` 패키지 (BaseLoss/ElementLoss/WeightingStrategy/LossFactory), trainer 연결 | Masked MSE/MAE/Huber, Weighted MSE 4종 실험 가능한 구조 확보 |
| STEP9.4 | Loss 비교 실험 | 위 4종 loss를 동일 backbone(ResNet18)에서 5-fold 비교 | Huber가 RMSE·R² 1위, MAE도 2위와 거의 동률(0.3275 vs 0.3245) → Huber 채택. 라벨수 역가중(exp4)은 4종 중 최하위 |
| STEP8 | HPO (Round1→Round2→Confirmation) | Optuna 베이지안 탐색, fold 1개로 33 trial 탐색 후 best params로 5-fold Confirmation 재실행 | `lr=2.37e-4, wd=7.97e-5, dropout=0.2, batch=64` 확정. Confirmation MAE 0.317~0.321(재실행 포함), R² 0.71~0.72 |
| STEP11 | 5-fold 앙상블 | Confirmation 5-fold 체크포인트 재생성 후 OOF 평가 + test set 5-model 평균 앙상블 | OOF property별 MAE/RMSE/R² 확보(Density R² 0.82 최고, Tc R² 0.585 최저), Kaggle 제출용 test 앙상블(3 샘플) 생성 |
| STEP12 | Grad-CAM 해석 | fold1 모델, layer4 대상 표준 Grad-CAM, property별 3샘플 | 약 절반은 화학적으로 타당한 구조(방향족 코어, 극성 연결부, 입체 장애 큰 치환기)에 반응, 나머지는 배경/모서리 아티팩트 |

전체 개선 추이(원본 단위 MAE/RMSE/R², z-score 아님):

| 단계 | 설정 | MAE | RMSE | R² |
|---|---|---|---|---|
| STEP7 baseline | ResNet18, MSE | 0.3380 | 0.5478 | 0.6874 |
| STEP9.4 loss 개선 | ResNet18, Huber | 0.3275 | 0.5357 | 0.7039 |
| STEP8 HPO 최종(8/11) | ResNet18, Huber, HPO 튜닝 | 0.3171 | 0.5209 | 0.7192 |
| STEP8 Confirmation 재실행(8/12) | 위와 동일 설정, 재현성 검증 | 0.3214 | 0.5279 | 0.7095 |

---

## 5. 숏페이퍼(학회 제출) 섹션별 배치 제안

- **Method / Experimental Setup**: STEP7~STEP9의 backbone·loss 선택 근거(표 형태로
  간단히), STEP8의 HPO 탐색 범위(Round1→Round2)와 GroupKFold(canonical_smiles)
  leakage 방지 설계를 서술.
- **Main Results 표**: 이 리포트 4번 "전체 개선 추이" 표(STEP7→STEP9.4→STEP8)를
  거의 그대로 사용 가능. 5-fold 평균만 넣고, 개별 fold 변동폭은 각주나 보충자료로.
- **Reproducibility 언급 (짧게)**: 8/11 vs 8/12 Confirmation 재실행 비교(1번 섹션)를
  "5-fold 평균은 안정적이나 개별 fold는 seed에 민감하다"는 한 문장 + 표 하나로
  본문 또는 appendix에 배치하면 심사자에게 좋은 신호가 될 수 있음.
- **Per-property 성능 표**: STEP11 OOF property별 표(2.1)를 메인 결과 표로 승격
  — z-score MAE보다 원본 단위 표가 독자에게 훨씬 직관적이고, 이 competition
  표준 평가 방식과도 일치할 가능성이 높음(overall 행은 넣지 말 것).
- **Discussion**: 2.2(라벨 개수와 R²가 단순 비례하지 않는다는 관찰)를 짧은
  분석 문단으로. "라벨이 많다고 성능이 자동으로 좋아지지 않으며, property의
  내재적 예측 난이도가 더 중요하다"는 메시지는 향후 데이터 수집 우선순위에도
  시사점을 줌.
- **Qualitative Analysis / Interpretability**: STEP12 Grad-CAM 중 가장 설득력
  있는 2~3장(FFV sample 2, Tg sample 18, Tc sample 4 추천)을 figure로 삽입하고,
  "일부 실패 케이스도 관찰됨"이라는 한계 문장을 함께 적어 과대 해석을 방지.
- **Limitations**: (a) fold 단위 재현성 노이즈, (b) OOF overall 지표의 스케일
  혼합 문제, (c) Grad-CAM이 배경/모서리에 반응하는 실패 케이스 — 이 세 가지를
  Limitations 절에 한 문단으로 묶어서 정직하게 명시하는 것을 권장.

---

## 6. 참고 — 재현에 사용된 파일/스크립트

- Confirmation 재실행: `hpo/run_confirmation.py` (변경 없음, 어제와 동일 config)
- 앙상블: `ensemble/ensemble_common.py`, `ensemble/oof_evaluation.py`,
  `ensemble/test_ensemble.py`
- Grad-CAM: `gradcam.py` (target layer: `ResNet18Backbone.feature_extractor[7]`
  = layer4)
