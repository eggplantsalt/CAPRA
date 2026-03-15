# tests/capra/ -- CAPRA unit tests
#
# Run with:  pytest tests/capra/ -v
#
# Pure-Python tests (no GPU / LIBERO required):
#   test_equivalence.py
#   test_metrics.py
#   test_footprint.py   (tests build_safety_target_distribution)
#   test_precursor.py
#   test_smoke_pipeline.py
#
# Phase 2 tests (require LIBERO + env):
#   test_snapshot.py
#   test_state_api.py
#   test_candidate_actions.py
