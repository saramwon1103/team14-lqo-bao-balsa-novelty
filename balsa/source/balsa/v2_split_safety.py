"""Fail-closed split membership checks used by official Balsa V2 runs."""
from pathlib import Path


def query_stems(paths):
    return {Path(value).stem for value in paths}


def assert_train_test_membership(expected_train, expected_test, actual_train, actual_test):
    expected_train=set(expected_train);expected_test=set(expected_test)
    actual_train=set(actual_train);actual_test=set(actual_test)
    if expected_train & expected_test:
        raise ValueError('V2 source manifest contains train/test overlap')
    if actual_train != expected_train:
        raise ValueError('Balsa training query membership differs from split_master')
    if actual_test != expected_test:
        raise ValueError('Balsa test query membership differs from split_master')
    if len(actual_train | actual_test) != 113:
        raise ValueError('V2 Balsa workload does not cover exactly 113 JOB queries')
