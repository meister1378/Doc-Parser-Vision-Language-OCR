# PaddleOCR-VL (LoRA) SFT 실행 가이드 (데이터 생성 → 학습 → 최종 모델 저장)

이 문서는 **OCR 데이터 생성(레이아웃 LMDB)**부터 **PaddleOCR-VL LoRA SFT 학습**, 그리고 **`ERNIE/LORA_MODEL_LOADANDSAVE.ipynb`로 최종 병합(merge) 저장**까지 한 번에 따라 실행할 수 있게 정리한 가이드입니다.

> 데이터 생성 파이프라인의 **데이터 구성/전처리(merged_json, lookup)** 까지는 기존 `ocr_test/OCR_PIPELINE/README.md`와 동일합니다.  
> 차이는 **`create_all_datasets_layout.py` 실행 시점부터**이며, 생성된 `*_layout.lmdb`를 이용해 본 repo(`ERNIE`)에서 SFT를 진행합니다.

---

## 0) 전제

- **레이아웃 LMDB 생성 스크립트 위치**: `ocr_test/OCR_PIPELINE/FAST/create_all_datasets_layout.py`
- **SFT 학습 실행 위치(권장)**: 이 문서가 있는 `ERNIE/` 디렉터리
- **학습 데이터(예시 경로)**: `/mnt/nas/ocr_dataset/*.lmdb`

---

## 1) (공통) 데이터 준비: merged_json / lookup

아래 단계는 기존 `ocr_test/OCR_PIPELINE/README.md`의 “2) merged_json 만들기”, “3) lookup 만들기”와 동일합니다.

- `merge_json_datasets.py`로 `*_merged.json` 준비
- `ftp_tree_viewer.py` / `convert_lookup_to_pickle.py`로 `lookup_*.pkl.gz` 준비

실행 위치/상대경로 이슈가 있으므로 **기존 문서 안내대로 `ocr_test/OCR_PIPELINE/FAST`에서 수행**하는 것을 권장합니다.

---

## 2) layout LMDB 생성 (`create_all_datasets_layout.py`)

`create_all_datasets_layout.py`를 실행해 **SFT 학습 입력으로 사용할 layout LMDB**를 생성합니다.

### 2-1) (선택) 환경변수

기존 파이프라인과 동일하게 레이아웃/테이블 관련 옵션을 환경변수로 받을 수 있습니다.

```bash
export FAST_DEBUG=0
export FAST_LAYOUT_DEVICE=gpu   # gpu|cpu
export FAST_TABLE_DEVICE=gpu    # gpu|cpu
export FAST_LAYOUT_MODEL=PP-DocLayoutV2
export FAST_TABLE_LAYOUT_THR=0.3
```

### 2-2) 실행

```bash
cd /home/mango/ocr_test/OCR_PIPELINE/FAST
python create_all_datasets_layout.py
```

### 2-3) 산출물(예)

아래처럼 `*_layout.lmdb`가 생성됩니다.

- `/mnt/nas/ocr_dataset/ocr_public_train_layout.lmdb`
- `/mnt/nas/ocr_dataset/public_admin_train_layout.lmdb`

---

## 3) PaddleOCR-VL LoRA SFT 학습 (erniekit)

학습 설정 파일은 아래 yaml을 사용합니다.

- `examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml`

`ERNIE/`에서 다음 명령어로 학습합니다.

```bash
cd /home/mango/ERNIE

CUDA_VISIBLE_DEVICES=0 erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
  model_name_or_path=PaddlePaddle/PaddleOCR-VL \
  train_dataset_path=./mnt/nas/ocr_dataset/ocr_public_train_layout.lmdb \
  eval_dataset_path=./mnt/nas/ocr_dataset/public_admin_train_layout.lmdb \
  output_dir=PaddleOCR-VL-SFT-OCRPUBLIC \
  gradient_accumulation_steps=2
```

> 참고: yaml 내부에도 기본값(예: `train_lmdb_paths`, `eval_dataset_path`, `output_dir`, `gradient_accumulation_steps`)이 들어있고, 위 커맨드는 이를 **런타임 override**합니다.

---

## 4) 최종 모델 저장(LoRA 병합) — `LORA_MODEL_LOADANDSAVE.ipynb`

학습이 끝나면 `output_dir` 아래에 `checkpoint-xxxxxx/`가 생성됩니다(예: `PaddleOCR-VL-SFT-OCRPUBLIC/checkpoint-250000`).

이 단계에서는 `ERNIE/LORA_MODEL_LOADANDSAVE.ipynb`를 사용해:

- 베이스 모델(`PaddlePaddle/PaddleOCR-VL`) + LoRA 체크포인트를 로드
- LoRA를 **merge**
- 최종 가중치를 **`model.safetensors`** 형태로 저장합니다(`safe_serialization=True`)

### 4-1) 노트북 설정(핵심 변수)

노트북 첫 셀에서 아래 두 값을 학습 결과에 맞게 지정합니다.

- `BASE = "PaddlePaddle/PaddleOCR-VL"`
- `LORA = "/home/mango/ERNIE/PaddleOCR-VL-SFT-OCRPUBLIC/checkpoint-250000"`  ← 본인 체크포인트 경로로 변경

### 4-2) 저장 동작

첫 셀을 실행하면 대략 아래 순서로 진행됩니다.

- `PaddleOCRVLForConditionalGeneration.from_pretrained(BASE, convert_from_hf=True)`
- `LoRAModel.from_pretrained(base_model, lora_path=LORA)`
- `lora_model.merge()` → `restore_original_model()`
- 병합된 모델을 `LORA` 경로에 `save_pretrained(..., safe_serialization=True)`로 저장

### 4-3) (필요 시) safetensors 전치(transpose) 보정

노트북 2번째 셀은 특정 키들의 weight가 전치되어야 하는 케이스를 위해 `model.safetensors`를 읽어 `_fixed.safetensors`로 저장하는 유틸입니다.

- 대상 파일: `.../checkpoint-xxxxxx/model.safetensors`
- 출력 파일: `.../checkpoint-xxxxxx/model_fixed.safetensors`
- 안내대로 `model_fixed.safetensors`를 `model.safetensors`로 교체해서 사용합니다.

---

## 5) Git ignore 권장(대용량 데이터/산출물/가중치)

이 repo에는 학습 산출물(`output_dir`, 체크포인트, 로그)과 대용량 데이터(LMDB, 이미지, 캐시)가 쉽게 쌓입니다.  
푸시 전에 `ERNIE/.gitignore`에 관련 패턴을 추가해 커밋에 포함되지 않도록 합니다(아래 `.gitignore`가 포함하도록 이미 반영되어야 합니다).

