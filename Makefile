.PHONY: test eval check smoke-real

test:
	python -m pytest

eval:
	python scripts/run_eval.py

check:
	powershell -ExecutionPolicy Bypass -File scripts/run_all_checks.ps1

smoke-real:
	powershell -ExecutionPolicy Bypass -File scripts/run_real_smoke.ps1
