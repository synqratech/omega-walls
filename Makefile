.PHONY: test eval eval-advanced check smoke-real demo-attack demo-benign

test:
	python -m pytest

eval:
	python -m omega eval --suite quick --strict

eval-advanced:
	python scripts/run_eval.py

check:
	powershell -ExecutionPolicy Bypass -File scripts/run_all_checks.ps1

smoke-real:
	powershell -ExecutionPolicy Bypass -File scripts/run_real_smoke.ps1

demo-attack:
	python -m omega demo attack

demo-benign:
	python -m omega demo benign
