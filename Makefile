.PHONY: validate checksums verify extract

validate:
	python scripts/validate_repository.py

checksums:
	python scripts/generate_checksums.py

verify:
	python scripts/verify_checksums.py

extract:
	python scripts/extract_archives.py
