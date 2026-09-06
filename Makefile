.PHONY: install test audit smoke train leaderboard predict validate

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

audit:
	python scripts/audit_dataset.py --data-root ../../geo_dataset

smoke:
	@echo "Use: python scripts/smoke_test.py --attachments PATH_TO_SAMPLES --labels PATH_TO_LABELS"

train:
	python scripts/train.py --config configs/efficientnet_b0_scratch.yaml --data-root ../../geo_dataset --fold 0 --device cuda

leaderboard:
	python scripts/cross_validate.py --config configs/efficientnet_b0_scratch.yaml --data-root ../../geo_dataset --device cuda

predict:
	python scripts/predict.py --checkpoint outputs/final/final_model.pt --data-root ../../geo_dataset --output predictions.csv --device cuda

validate:
	python scripts/validate_submission.py --predictions predictions.csv --holdout-dir ../../geo_dataset/holdout_public
