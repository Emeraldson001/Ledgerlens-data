"""CI entry point: fail if any registered importer is non-conformant."""

import sys

from ingestion import importer_registry, registered_importers  # noqa: F401

failures = importer_registry.verify_all()
for f in failures:
    print("FAIL:", f)
print(f"{len(importer_registry.registered())} importer(s) checked, {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
