.PHONY: test eval check smoke-real demo bench bench-full secret-scan source-archive

test:
	python -m pytest

eval:
	python scripts/run_eval.py

check:
	powershell -ExecutionPolicy Bypass -File scripts/run_all_checks.ps1

smoke-real:
	powershell -ExecutionPolicy Bypass -File scripts/run_real_smoke.ps1

demo:
	python scripts/quick_demo.py --mode pi0

bench:
	python scripts/check_benchmarks_contract.py --smoke-exec

bench-full:
	python scripts/run_benchmarks_full.py

secret-scan:
	python scripts/secret_scan.py .

source-archive: secret-scan
	python scripts/build_clean_source_archive.py
