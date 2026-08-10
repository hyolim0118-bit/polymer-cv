# 2026-08-10 작업 검토 리포트

학회 short paper 제출(마감 8/17)을 앞두고 진행한 GroupKFold leakage 수정,
STEP7(backbone 비교), STEP9(Loss Architecture) 작업을 파일 기준으로 검토하고,
STEP8(HPO) 착수 전 정리가 필요한 항목을 정리한다.

## 오늘 한 일 (시간 순)

1. GitHub 첫 백업 + `.gitignore` 정리 (`data/raw`, `images`, `checkpoints`, `*.pth` 등 제외)
2. GroupKFold data leakage 재검증 — `canonical_smiles` 기준 실제 중복 2개 그룹 발견,
   과거 5-fold 결과가 5개 fold 중 3개(fold 2, 3, 5)에서 실제 leakage 있었음을 재현 검증으로 확인,
   `cross_validation.py`의 그룹 키를 `canonical_smiles`로 최종 확정
3. STEP7: Backbone 비교 (ResNet18 vs EfficientNet-B0), GPU(Vast.ai RTX 4090)에서 5-fold씩 실행
4. STEP9: Loss Architecture 구현(`losses/` 패키지: BaseLoss, ElementLoss, WeightingStrategy, LossFactory) + trainer 연결
5. STEP9.4: Loss 4개(Masked MSE/MAE/Huber, Weighted MSE) 비교 실험

---

## 1. STEP7 / STEP9 반영 여부 — 파일 기준 검토

| 항목 | 상태 | 확인 내용 |
|---|---|---|
| `losses/` 패키지 | ✅ 정상 | `base.py`, `elements/{mse,mae,huber}.py`, `weighting/{uniform,static,uncertainty}.py`, `factory.py` 모두 존재 (uncertainty.py는 설명엔 없었지만 추가로 있음) |
| `training/trainer.py` → LossFactory | ✅ 정상 | `from losses.factory import LossFactory` + `self.criterion = LossFactory.build(config, property_names=...)` — 2줄 변경 확인 |
| `training/config.yaml` loss 섹션 | ✅ 정상 | `element_loss` / `weighting` / `reduction` nested 구조로 변경됨 |
| `models/efficientnet.py` + `factory.py` 등록 | ✅ 정상, 실제 사용됨 | `models/factory.py`의 `_BACKBONE_REGISTRY`에 `resnet18`/`efficientnet_b0` 등록. `run_resnet18.py`/`run_efficientnet.py`가 이 factory로 실제 5-fold를 돌렸고 결과 파일도 존재 |
| `training/training real/cross_validation.py`의 canonical_smiles GroupKFold | ✅ 유지됨 | `groups=pool_df["canonical_smiles"]`, leakage assert가 이후 커밋에서도 그대로 살아있음 |

### 검토 중 추가로 발견한 문제

- **`training/training real/trainer.py`가 옛날 loss 코드에 머물러 있음.** LossFactory 전환은 top-level `training/trainer.py`에만 적용됐고, `training/training real/trainer.py`는 여전히 `from losses.loss import build_loss`(구식)를 쓴다. 현재 실행 경로(`run_resnet18.py` 등은 top-level `training.trainer.Trainer`를 import)에서는 문제 없지만, 나중에 "training real" 폴더 안 파일을 진짜인 줄 알고 고치면 옛날 loss로 되돌아가는 함정이 된다.
- **`.gitignore`가 깨져 있음.** 실제 파일을 열어보면 `cat > .gitignore << 'EOF'` 줄과 마지막 `EOF` 줄이 패턴이 아니라 파일 내용 그 자체로 들어가 있다 (heredoc이 셸에서 해석되지 않고 그대로 파일에 써짐). git이 `EOF`라는 이름의 파일/폴더만 없으면 무해하지만 의도한 대로 동작하는 게 아니므로 정리 필요.
- **`models/factory.py`(flat) 옆에 `models/backbone/` + `models/model.py`라는 병렬 구현이 하나 더 있음.** `models/model.py`는 자기 docstring에 "STEP7 마이그레이션의 핵심 변경 파일"이라 써 있지만, 실제로 STEP7 backbone 비교를 돌린 건 이 파일이 아니라 `models/factory.py`(flat) 쪽이다. `models/backbone/`는 `resnet34/50`, `efficientnet_b1`, `freeze_backbone`, Grad-CAM 훅까지 더 완성도 있게 짜여 있지만 어디에도 실제로 연결돼 있지 않다 — 즉 검증된 적 없는 코드다. (4번 항목에서 이어서 설명)

---

## 2. Loss 비교 결과 요약 (`results/step9_4/loss_comparison.csv`)

| 실험 | val_loss_mean | MAE | RMSE | R² |
|---|---|---|---|---|
| exp1 Masked MSE | 0.239 | 0.3295 | 0.5365 | 0.7002 |
| exp2 Masked MAE | 0.257 | **0.3245** | 0.5399 | 0.6975 |
| exp3 Masked Huber | 0.075 | 0.3275 | **0.5357** | **0.7039** |
| exp4 Weighted MSE | 0.277 | 0.3397 | 0.5433 | 0.6938 |

**주의**: `val_loss_mean`으로 순위를 매기면 안 된다. MSE/MAE/Huber는 손실 함수 자체의 스케일이 달라서(Huber는 δ=1 안에서만 이차항이라 값이 구조적으로 작게 나옴) exp3의 0.075가 "압도적으로 좋다"는 건 착시다. 공정한 비교는 원래 단위로 환산된 **MAE/RMSE/R²** 세 지표다.

이 세 지표로 보면 **Huber(exp3)가 RMSE·R²에서 1위, MAE에서도 최고(exp2)와 거의 차이 없음**(0.3275 vs 0.3245). MAE를 직접 최적화한 exp2가 MAE 지표에서만 근소 우위인 건 자기 지표를 직접 최적화한 결과라 큰 의미는 없다. **exp4(라벨 개수 역수 가중 MSE)는 세 지표 모두 최하위** — 라벨이 적은 property(아마 Tc)에 과가중한 게 오히려 역효과였을 가능성.

**결론: Huber loss(exp3_masked_huber)를 STEP8 HPO 기준 loss로 가져가는 게 타당해 보임.** 다만 fold 간 std(MAE_std ~0.026~0.032)가 실험 간 차이(~0.01~0.015)보다 커서, 이 순위는 "확실히 유의미하다"기보다는 "현재까지 가장 근거 있는 선택"에 가깝다.

### 참고: STEP7 Backbone 비교 (재계산 확인)

| Backbone | MAE (5-fold 평균) | RMSE | R² |
|---|---|---|---|
| ResNet18 | 0.3380 | 0.5478 | 0.6874 |
| EfficientNet-B0 | 0.3423 | 0.5512 | 0.6854 |

차이가 각각 0.004 / 0.003 / 0.002 수준으로, loss 비교 실험에서 관측된 fold-std(~0.03)보다 훨씬 작다. "ResNet18이 근소 우세, 통계적으로 유의미한 차이는 아님"이라는 기존 판단은 재계산으로도 확인됨.

---

## 3. `models/config.yaml` / `training/training real/config.yaml`

**둘 다 죽은 파일. 삭제해도 됨.**

- 저장소의 어떤 `.py` 파일도 이 두 경로를 직접 로드하지 않음 (grep으로 전수 확인).
- `run_resnet18.py`, `run_efficientnet.py`, `run_loss_experiments.py`, `hpo/run_hpo.py`, `training/train.py`, `training/smoke_test.py` 전부 **top-level `training/config.yaml`** 하나만 기본값으로 사용 — "training real" 폴더 안의 스크립트들조차 자기 폴더의 config.yaml이 아니라 top-level 걸 가리킴.
- `models/config.yaml`은 `training/config.yaml`의 앞부분(data/split/image/augmentation/dataloader/labels/model)만 그대로 복사해놓은 형태로, 초기 스캐폴딩 잔재로 추정.

---

## 4. STEP8 HPO 전에 정리/수정할 것

### 🔴 최우선 — 지금 고치지 않으면 STEP8에서 leakage 버그 재발

`validation/` 패키지 전체(`leakage_check.py`, `manager.py`, `strategies/group_kfold.py`)와 이를 쓰는 `run_cv.py`(`GROUP_COL = "SMILES"`), `hpo/search.py`, `hpo/search_space.py`, `hpo/confirm.py`가 **여전히 raw `SMILES`를 그룹 키로 사용**한다. `cross_validation.py`에만 canonical_smiles 수정을 적용했고, 이 HPO용 validation 스택은 손대지 않았다. `hpo/search.py`의 "Search Stage"가 이 `validation.strategies`를 그대로 재사용하도록 설계돼 있으므로(주석에도 명시), 이대로 STEP8을 시작하면 지난번 발견한 그 2개 중복 분자가 다시 train/valid에 걸쳐 들어갈 수 있다.

→ **`run_cv.py`의 `GROUP_COL`과 관련 그룹 키를 `canonical_smiles`로 바꾸는 작업을 STEP8 착수 전에 먼저 할 것.**

### 🟠 두 번째 — 어느 모델 팩토리가 "진짜"인지 결정 필요

`models/factory.py`(flat, 실제 STEP7 검증에 쓰인 것)와 `models/backbone/` + `models/model.py`(더 잘 만들어졌지만 검증된 적 없는 것)가 공존한다. `hpo/run_hpo.py`(실제 실행 가능한 HPO 스크립트)는 전자를 쓰고, `hpo/search.py`(범용 엔진, train_fn 미배선)의 설계 의도는 후자를 가정하고 있는 것으로 보인다.

→ STEP8에서 어느 쪽 위에 지을지 먼저 정하고, 안 쓰는 쪽은 삭제하거나 최소한 "미사용/실험적" 표시를 해둘 것.

### 🟡 세 번째 — `training/training real/` 폴더 정리

이 폴더는 이제 `cross_validation.py` 하나만 실제로 쓰인다 (경로에 공백이 있어 `importlib.util.spec_from_file_location`으로 억지로 불러오는 방식). 나머지(`trainer.py`, `train.py`, `validate.py`, `smoke_test.py`, `config*.yaml`, `checkpoints/`, `dry_run_checkpoints/`, `smoke_checkpoints/`)는 top-level `training/`의 오래된 복사본이라 안 쓰인다 (위 1번 항목의 옛날 loss 코드가 그 증거).

→ STEP8에서 파일이 더 늘어나기 전에, `cross_validation.py`를 top-level `training/`으로 옮겨 공백 경로 문제를 없애고 나머지 사본은 삭제할 것.

### 자잘한 것들

- `.gitignore`의 heredoc 잔재(`cat > .gitignore << 'EOF'`, `EOF` 줄) 정리
- `losses/loss.py`(구 `build_loss`)는 위 세 번째 항목을 정리하면 완전히 죽은 코드가 됨 — 같이 삭제
- `run_loss_experiments_local_backup.py` (untracked) — 의도한 백업인지 확인 후 커밋하거나 삭제
- `results/` 하위가 `result/`, `baseline/`, `merged/`, `result/merged_before_kfold` 등 임시 이름으로 뒤섞여 있음 — `results/step9_4/` 같은 일관된 규칙을 STEP8 trial 결과에도 적용하기 전에 한 번 정리할 것

---

## 부록: GroupKFold Leakage 재현 검증 상세

- `data/processed_merged/processed_metadata.csv`의 train pool(8,974개) SMILES를 `Chem.CanonSmiles()`로 정규화한 결과, 문자열은 다르지만 같은 분자인 쌍이 2그룹(4행) 발견됨 (`id 3147438427`/`3147438428`, `id 3147438423`/`3147438580`, 둘 다 방향환 표기 순서 차이).
- `results/result/kfold_results.csv`(R² 0.66~0.78)를 만든 과거 실행 조건(`config_merged.yaml`: n_splits=5, shuffle=true, seed=42)을 그대로 재현한 결과, **5개 fold 중 3개(fold 2, 3, 5)에서 실제로 leakage 발생**이 확인됨.
- 영향 규모: fold당 최대 1쌍(valid ~1,795행 중 1행)뿐이라 MAE/RMSE/R² 등 집계 지표에 미친 실질적 영향은 통계적으로 무시할 수준일 가능성이 높으나, "leakage 없음"이라고 논문에 쓸 수는 없었던 상태.
- `canonical_smiles` 기준 GroupKFold로 교체 후 동일 조건 재검증 → 5개 fold 모두 겹침 0건.
