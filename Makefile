# Deprecated: use ./run.sh (GNU make is not required).
# Kept as a thin pointer for muscle memory.

.PHONY: setup gen-data mock eval-all eval-cer eval-forgery eval-face eval-tc6 report freeze

setup gen-data mock face calibrate-face test-face eval-all eval-cer eval-forgery eval-face eval-tc6 report freeze:
	./run.sh $@
