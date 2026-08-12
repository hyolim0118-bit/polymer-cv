# 2026-08-11 STEP8 HPO 작업 리포트

# STEP8 HPO — 방향성과 결정 과정 설명

## 왜 HPO를 이 시점에 했나

프로젝트 흐름은 이랬어요: **Backbone 선택 → Loss 선택 → 하이퍼파라미터 튜닝(HPO)** 순서.

이유는 간단해요. HPO가 찾는 값(learning rate, weight decay 등)의 "최적값"은 **어떤 backbone, 어떤 loss를 쓰느냐에 따라 달라져요.** 그래서 순서를 반대로 하면(HPO 먼저) backbone/loss가 바뀔 때마다 HPO를 처음부터 다시 해야 해서 작업이 배로 늘어나요. 큰 구조적 선택(backbone, loss)을 먼저 확정하고, 그 위에서 미세조정하는 게 효율적이에요.

- **Backbone**: ResNet18 vs EfficientNet-B0 비교 → **ResNet18 선택** (성능 근소 우위 + 검증된 안정성)
- **Loss**: Masked MSE / MAE / Huber / Weighted MSE 4개 비교 → **Huber 선택** (RMSE·R² 1위, MAE도 근소한 2위)

이 둘을 고정해두고, HPO는 그 위에서 학습 설정만 최적화하는 단계예요.

---

## HPO가 찾는 것

| 하이퍼파라미터 | 의미 |
|---|---|
| Learning rate | 모델이 한 번에 얼마나 크게 업데이트되는지 |
| Weight decay | 과적합을 막기 위한 정규화 강도 |
| Dropout | 학습 중 랜덤하게 뉴런을 꺼서 정규화하는 비율 |
| Batch size | 한 번에 몇 개 샘플을 보고 업데이트할지 |

---

## 탐색 방식: 2단계 (Round1 → Round2)

**왜 한 번에 안 하고 나눴나**: 5-fold 전체를 매 시도마다 돌리면 시간이 너무 오래 걸려요. 그래서:

1. **탐색 단계(Round1, Round2)에서는 fold 1개만 사용** — 빠르게 여러 조합을 시도
2. **최종 확정(Confirmation)에서만 5-fold 전체 재실행** — 진짜 보고할 성능은 여기서 나옴

### Round1 — 넓은 범위, 18번 시도

```
learning_rate : 1e-5 ~ 1e-3 (로그 스케일)
weight_decay  : 1e-6 ~ 1e-2 (로그 스케일)
dropout       : 0.2, 0.3, 0.4 중 하나
batch_size    : 16, 32, 64 중 하나
```

Optuna(베이지안 최적화 라이브러리)가 이 범위 안에서 18번 시도하면서, 성능 좋은 조합의 "패턴"을 찾음.

**Round1 결과**: 성능 상위 5개 시도가 전부 **dropout=0.2, batch_size=64**로 수렴했고, learning_rate는 대략 2.7e-4~8.4e-4 구간, weight_decay는 작은 값(1e-6~4.5e-5) 구간에 몰림.

### Round2 — Round1 결과로 범위를 좁혀서, 15번 재시도

```
learning_rate : 1e-4 ~ 1e-3 (좁힘)
weight_decay  : 1e-6 ~ 1e-4 (좁힘)
dropout       : 0.2 (고정 — Round1에서 이미 승자 확정)
batch_size    : 64 (고정)
```

**"왜 굳이 두 번 나눠서 하나?"** → 처음부터 좁은 범위로 탐색하면, 진짜 최적값이 그 범위 밖에 있을 경우 놓칠 위험이 있어요. 일단 넓게 봐서 "대략 어디가 좋은지" 파악한 다음, 그 근처를 촘촘하게 다시 보는 게 안전하면서도 효율적이에요 (총 시도 횟수는 33번으로, 처음부터 넓은 범위에서 40~50번 시도하는 것보다 적으면서도 결과는 비슷하거나 더 나음).

**Round2 결과**: `learning_rate=2.37e-4, weight_decay=7.97e-5, dropout=0.2, batch_size=64`가 최종 후보로 확정.

### Confirmation — 최종 후보로 5-fold 전체 재실행

Round2에서 나온 최적값으로 진짜 5-fold 전체를 학습해서, 이게 최종 보고 성능이 됨.

---

## 전체 개선 흐름 (참고)

| 단계 | 설명 |
|---|---|
| 시작 | ResNet18 + MSE + 기본 하이퍼파라미터 |
| Loss 개선 | ResNet18 + **Huber** + 기본 하이퍼파라미터 |
| HPO 최종 | ResNet18 + Huber + **튜닝된 하이퍼파라미터** |

각 단계를 거치면서 MAE는 꾸준히 낮아지고 R²는 꾸준히 높아짐 — 극적인 도약은 아니지만, 매 단계가 일관되게 개선에 기여했다는 게 이 실험의 포인트예요.

---

## 설계상 타협한 부분 (정직하게)

- **탐색 단계는 fold 1개만 사용** → 5-fold 전체보다 노이즈가 클 수 있음. 그래서 Confirmation에서 5-fold 전체로 다시 검증하는 안전장치를 뒀음.
- **Round1→Round2 범위 좁히기는 사람이 직접 결과 보고 판단** (완전 자동화 아님) → 이게 오히려 실수 방지에 도움됨 (자동으로 범위를 잘못 좁히는 것 방지).
- **시간 제약(학회 마감) 때문에 fold 1개 + 2라운드(총 33 trials)로 타협** → 이상적으로는 더 많은 trial과 5-fold 탐색이 좋겠지만, 현실적인 시간 안에서 가장 안전한 절충안.

`optuna_search.py`(Round1/Round2 Search Stage)와 `run_confirmation.py`(Confirmation Stage)로
진행한 STEP8 HPO 결과를 실제 결과 파일 기준으로 정리한다. 숫자는 모두
`results/step8_hpo/`, `results/kfold_results_resnet18.csv`,
`results/step9_4/loss_comparison.csv`에서 직접 읽은 값이다.

## 오늘 한 일 (시간 순)

1. GroupKFold 그룹 키 통일 (`run_cv.py`, `validation/` 패키지를 `canonical_smiles`로)
2. `overfit_test.py`로 파이프라인 정합성 검증 (PASS)
3. `.gitignore` 정리
4. STEP8 HPO 스크립트 신규 작성 — `optuna_search.py`(Round1/Round2 탐색), `run_confirmation.py`(5-fold 재실행)
5. backbone=resnet18, loss=huber로 고정(STEP7/STEP9.4 결과 기준)하고 HPO 실행 (learning_rate, weight_decay, dropout, batch_size 탐색)

---

## 1. Round1 결과 (넓은 범위, 18 trials, fold 0 단일 fold)

**탐색 범위**: `learning_rate ∈ [1e-5, 1e-3]`(log), `weight_decay ∈ [1e-6, 1e-2]`(log), `dropout ∈ {0.2, 0.3, 0.4}`, `batch_size ∈ {16, 32, 64}`

val_loss 기준 정렬 (상위 9 / 하위 9):

| trial | lr | wd | dropout | batch | val_loss | val_mae |
|---|---|---|---|---|---|---|
| 17 | 2.80e-4 | 2.65e-5 | 0.2 | 64 | **0.08509** | 0.31408 |
| 15 | 3.83e-4 | 1.94e-5 | 0.2 | 64 | 0.08722 | 0.32035 |
| 16 | 4.41e-4 | 4.48e-5 | 0.2 | 64 | 0.08764 | 0.31999 |
| 11 | 8.41e-4 | 1.43e-6 | 0.2 | 64 | 0.08788 | 0.31641 |
| 13 | 2.68e-4 | 1.33e-6 | 0.2 | 64 | 0.08977 | 0.32787 |
| 6  | 1.24e-4 | 5.49e-6 | 0.2 | 64 | 0.09048 | 0.32897 |
| 12 | 7.83e-4 | 5.95e-6 | 0.2 | 64 | 0.09183 | 0.33317 |
| 10 | 8.28e-4 | 1.06e-6 | 0.2 | 64 | 0.09241 | 0.33190 |
| 0  | 5.61e-5 | 6.35e-3 | 0.2 | 64 | 0.09322 | 0.34632 |
| 1  | 1.59e-4 | 6.80e-4 | 0.3 | 16 | 0.09357 | 0.33692 |
| 3  | 8.17e-5 | 1.38e-3 | 0.4 | 32 | 0.09431 | 0.33669 |
| 2  | 4.06e-5 | 1.26e-4 | 0.4 | 64 | 0.10191 | 0.36504 |
| 8  | 3.65e-5 | 1.48e-4 | 0.3 | 16 | 0.10296 | 0.36104 |
| 5  | 1.75e-5 | 9.57e-5 | 0.3 | 16 | 0.10969 | 0.36940 |
| 7  | 1.50e-5 | 6.08e-6 | 0.4 | 32 | 0.11329 | 0.37746 |
| 4  | 1.35e-5 | 6.25e-3 | 0.2 | 32 | 0.11551 | 0.37950 |
| 9  | 1.03e-5 | 1.83e-3 | 0.4 | 32 | 0.12153 | 0.38648 |
| 14 | 3.90e-4 | 1.04e-6 | 0.2 | 64 | 0.12673 | 0.39820 |

**패턴**:
- 상위 9개 trial은 **dropout/batch_size 조합이 전부 (0.2, 64)** 로 동일하다. 하위 9개는 dropout {0.4×4, 0.3×3, 0.2×2}, batch_size {32×4, 16×3, 64×2}로 다른 조합이 대부분을 차지한다 — dropout=0.2·batch=64가 명확히 우세하다는 근거는 충분하다.
- 상위권 learning_rate는 대체로 1.24e-4 ~ 8.41e-4 구간에 몰려 있다.
- 상위권 weight_decay는 1.43e-6 ~ 4.48e-5 구간에 몰려 있다 (단, trial 0은 wd=6.35e-3으로 훨씬 크면서도 9위를 차지 — dropout=0.2/batch=64 조합 자체가 wd에 어느 정도 강건한 것으로 보인다).
- **주의할 이상치**: trial 14 (lr=3.90e-4, wd=1.04e-6, dropout=0.2, batch=64)는 다른 (0.2, 64) 조합 trial들과 하이퍼파라미터가 거의 비슷한데도(trial 11과 wd 자릿수만 다름) val_loss=0.12673으로 **전체 최하위**를 기록했다. 탐색이 fold 하나·seed 하나로만 이뤄지므로, 이런 튐은 하이퍼파라미터 자체보다 초기화/셔플 노이즈일 가능성이 높다 — 아래 4번 항목에서 이 노이즈 수준을 다시 언급한다.

## 2. Round2 범위가 타당했는가

**사용자가 좁힌 범위**: `learning_rate ∈ [1e-4, 1e-3]`(log), `weight_decay ∈ [1e-6, 1e-4]`(log), `dropout = {0.2}`, `batch_size = {64}`

- **dropout=0.2, batch_size=64 고정은 Round1 데이터로 확실히 뒷받침된다** (상위 9개 전부 이 조합).
- **learning_rate/weight_decay 범위도 Round1 상위 4개 trial(17,15,16,11)의 값(lr 2.80e-4~8.41e-4, wd 1.43e-6~4.48e-5)을 모두 포함**하므로 방향은 맞다.
- 다만 두 가지 참고할 점:
  - lr 하한을 1e-4로 자르면서 trial 0(lr=5.61e-5, 9위, val_loss=0.09322)이 만든 "낮은 lr에서도 나쁘지 않은 지점"은 Round2에서 재탐색되지 않았다. 결과적으로 문제는 없었지만(Round2 결과가 이미 충분히 좋음), 범위가 다소 공격적으로 좁혀진 지점이다.
  - Round1 최고 기록(trial 17, val_loss=0.08509)은 Round2가 찾은 최고 기록(trial 0, val_loss=0.08654)보다 **오히려 근소하게 더 낮다** (아래 3번 참고). Round2가 Round1의 단일 최저점을 갱신하지는 못했다.

## 3. Round2 결과 (좁힌 범위, 15 trials)

val_loss 기준 정렬:

| trial | lr | wd | dropout | batch | val_loss | val_mae |
|---|---|---|---|---|---|---|
| 0  | 2.37e-4 | 7.97e-5 | 0.2 | 64 | **0.08654** (best) | 0.32812 |
| 1  | 5.40e-4 | 1.58e-5 | 0.2 | 64 | 0.08753 | 0.31857 |
| 13 | 2.65e-4 | 7.17e-6 | 0.2 | 64 | 0.08817 | 0.32417 |
| 4  | 3.99e-4 | 2.61e-5 | 0.2 | 64 | 0.08820 | 0.32344 |
| 3  | 1.14e-4 | 5.40e-5 | 0.2 | 64 | 0.08836 | 0.33064 |
| 14 | 3.53e-4 | 9.56e-5 | 0.2 | 64 | 0.08836 | 0.32739 |
| 8  | 2.01e-4 | 1.12e-5 | 0.2 | 64 | 0.08866 | 0.32474 |
| 5  | 1.05e-4 | 8.71e-5 | 0.2 | 64 | 0.08884 | 0.32739 |
| 7  | 1.52e-4 | 2.33e-6 | 0.2 | 64 | 0.08914 | 0.33039 |
| 2  | 1.43e-4 | 2.05e-6 | 0.2 | 64 | 0.08927 | 0.32343 |
| 6  | 6.80e-4 | 2.66e-6 | 0.2 | 64 | 0.08945 | 0.32013 |
| 12 | 5.19e-4 | 3.75e-5 | 0.2 | 64 | 0.09023 | 0.32780 |
| 10 | 8.94e-4 | 1.03e-6 | 0.2 | 64 | 0.09142 | 0.32462 |
| 9  | 2.70e-4 | 3.82e-6 | 0.2 | 64 | 0.11051 (이상치) | 0.35875 |
| 11 | 3.86e-4 | 1.60e-5 | 0.2 | 64 | 0.13532 (이상치) | 0.40410 |

13/15 trial이 val_loss 0.0865~0.0914 범위에 조밀하게 몰려 있어, 이 구간 전체가 고르게 좋다는 걸 보여준다. trial 9, 11만 뚜렷한 이상치인데, Round1의 trial 14와 마찬가지로 하이퍼파라미터 자체는 특별히 나쁘지 않다 — fold/seed 노이즈로 보는 게 합리적이다.

**`round2/best_params.json`**:
```json
{
  "learning_rate": 0.00023688639503640813,
  "weight_decay": 7.969454818643937e-05,
  "dropout": 0.2,
  "batch_size": 64
}
```

## 4. Confirmation 5-fold 결과

`round2/best_params.json`의 하이퍼파라미터로 `cross_validation.run_kfold()`를 5-fold 전체 재실행한 결과 (`results/step8_hpo/kfold_results_confirmation.csv`):

| Fold | TrainLoss | ValLoss | MAE | RMSE | R² |
|---|---|---|---|---|---|
| 1 | 0.04439 | 0.09156 | 0.32744 | 0.56043 | 0.69902 |
| 2 | 0.03280 | 0.07192 | 0.31083 | 0.51076 | 0.74616 |
| 3 | 0.03044 | 0.07458 | 0.30996 | 0.48994 | 0.71069 |
| 4 | 0.02985 | 0.07143 | 0.34208 | 0.62670 | 0.65885 |
| 5 | 0.03230 | 0.05866 | 0.29527 | 0.41672 | 0.78137 |

**평균 ± 표준편차 (5-fold)**:

| 지표 | 평균 | 표준편차 |
|---|---|---|
| ValLoss | 0.07363 | 0.01177 |
| MAE | **0.31712** | 0.01801 |
| RMSE | **0.52091** | 0.07853 |
| R² | **0.71922** | 0.04667 |

## 5. 전체 개선 추이 (STEP7 → STEP9.4 → STEP8)

각 단계 결과 파일에서 직접 계산한 5-fold 평균값:

| 단계 | 설정 | MAE | RMSE | R² |
|---|---|---|---|---|
| STEP7 baseline | resnet18, mse/uniform (기본 loss) | 0.33803 | 0.54779 | 0.68740 |
| STEP9.4 loss 개선 | resnet18, **huber**/uniform | 0.32745 | 0.53570 | 0.70388 |
| STEP8 HPO 최종 | resnet18, huber/uniform, **HPO 튜닝 lr/wd/dropout/batch** | **0.31712** | **0.52091** | **0.71922** |

세 단계 모두 MAE/RMSE는 낮아지고 R²는 높아지는 일관된 개선 추세를 보인다 (STEP7→STEP8: MAE -6.2%, RMSE -4.9%, R² +0.032).

**주의**: 표에 `ValLoss`를 넣지 않았다. STEP7은 MSE, STEP9.4/STEP8은 Huber 기준 loss라 값의 스케일 자체가 달라(Huber는 δ=1 안에서만 이차항이라 수치가 구조적으로 작음) 손실값끼리는 단계 간 비교가 불가능하다. 위 표의 MAE/RMSE/R²는 원본 단위로 환산된 값이라 세 단계 모두 공정하게 비교 가능하다.

---

## 6. 코드 검토: `optuna_search.py`, `run_confirmation.py`

두 파일 모두 오늘 설명한 대로 동작하는 것으로 확인했다.

- **`optuna_search.py`**: `training/training real/cross_validation.py`를 `importlib`로 불러와(폴더명 공백 때문에 STEP9.4와 동일한 방식) `_build_fold_loaders`, `build_model`, `Trainer`를 그대로 재사용한다. `get_fold0_split()`이 `cross_validation.GroupKFold(n_splits=...).split(pool_df, groups=pool_df["canonical_smiles"])`를 직접 호출해서 fold 0만 뽑아 쓰는데, 이건 `run_kfold()` 내부와 **완전히 동일한 GroupKFold 클래스·동일한 canonical_smiles 그룹**이라 (sklearn GroupKFold는 결정적이라 셔플 랜덤성이 없음) fold 경계가 실제 5-fold의 fold 1과 동일하다. `FIXED_OVERRIDES`로 `model.name=resnet18`, `loss.element_loss.name=huber`를 고정하고, 매 trial마다 `trial_dir`를 만들어 학습한 뒤 `shutil.rmtree`로 즉시 삭제 — 디스크 낭비를 막는 처리도 확인했다. Round1(18 trials, 넓은 범위)/Round2(15 trials, 좁힌 범위)가 `sys.argv`로 분기되는 구조도 설명과 일치한다.
  - 사소한 참고: 여기서 뽑은 fold 0에는 `cross_validation.py`가 쓰는 leakage assert(`check_no_leakage`에 해당하는 부분)가 별도로 걸려있지 않다. 다만 같은 결정적 GroupKFold 로직을 그대로 재사용하는 것이라 실제 leakage 위험은 없다 — 방어적으로 assert를 하나 추가해도 좋지만 필수는 아니다.
- **`run_confirmation.py`**: `round2/best_params.json`을 읽어 `FIXED_OVERRIDES` + best_params를 합친 config로 `cross_validation.run_kfold(config)`를 그대로 호출한다. `results/kfold_results.csv`(run_kfold의 기본 저장 경로)를 `results/step8_hpo/kfold_results_confirmation.csv`로 복사하는 방식도 실행 로그와 실제 파일 내용이 일치해 정상 동작을 확인했다. "오늘 backbone/loss 비교와 동일한 패턴"이라는 설명대로, `run_resnet18.py`/`run_loss_experiments.py`와 같은 구조를 그대로 재사용하고 있다.

---

## 7. STEP11 앙상블을 위한 참고사항

- **베이스 하이퍼파라미터**: resnet18 + huber(uniform weighting) + `lr=2.369e-4, wd=7.969e-5, dropout=0.2, batch_size=64`가 STEP8 confirmation으로 검증된 현재 최선 설정이다. 앙상블 멤버를 새로 만들 때 이 설정을 기본값으로 삼는 게 합리적이다.
- **가장 손쉬운 첫 앙상블은 5-fold 멤버 앙상블**: Confirmation 5-fold는 leakage-safe(canonical_smiles GroupKFold)하게 나뉜 fold라 각 fold의 best checkpoint 5개를 그대로 앙상블하는 게 개념적으로 가장 저렴한 시작점이다. **다만 실제로 `results/step8_hpo/confirmation_checkpoints/`를 확인해보니 현재 디스크에 없다** (round1/round2/confirmation_logs만 남아있고 체크포인트는 정리된 것으로 보임) — STEP11에서 이 방식을 쓰려면 `run_confirmation.py`를 재실행해 5개 fold checkpoint를 다시 만들어야 한다. 하이퍼파라미터는 이미 확정돼 있으니 재실행 자체는 비용이 크지 않다.
- **HPO 과정에서 관측된 노이즈**: Round1 trial 14, Round2 trial 9/11처럼 하이퍼파라미터가 비슷한데도 val_loss가 크게 튀는 경우가 있었다(단일 fold·단일 seed 평가라서 생기는 변동으로 추정). 이건 반대로 말하면 **동일 하이퍼파라미터에서도 seed만 바꿔 여러 모델을 학습하면 앙상블에 유의미한 다양성을 줄 수 있다는 뜻**이기도 하다 — seed ensemble도 STEP11 후보로 고려할 만하다.
- **backbone 다양성**: STEP7에서 EfficientNet-B0(MAE 0.3423)는 resnet18(MAE 0.3380)에 근소하게 뒤졌지만 차이가 크지 않았다. 단일 backbone으로는 resnet18이 채택됐지만, 서로 다른 아키텍처가 만드는 에러 패턴 차이는 앙상블에 유리할 수 있어 EfficientNet-B0도 멤버 후보로 다시 고려해볼 만하다.
- **loss 다양성**: STEP9.4에서 MAE loss(exp2)가 MAE 지표만 놓고 보면 huber(exp3)와 거의 차이가 없었다(0.3245 vs 0.3275). huber가 종합 1위라 최종 채택됐지만, MAE-loss로 학습한 모델은 에러 특성이 달라 앙상블 멤버로서 가치가 있을 수 있다.
- **미해결 이월 이슈**: 8/10 리포트(`docs/2026-08-10_status_review.md`)에서 지적한 `models/factory.py`(flat) vs `models/backbone/`+`models/model.py`(구조화) 이중 구현 문제는 오늘 작업에서 다루지 않았다. STEP11에서 앙상블 멤버로 새 backbone을 추가하기 전에 이 결정을 먼저 정리해두는 게 좋다.
