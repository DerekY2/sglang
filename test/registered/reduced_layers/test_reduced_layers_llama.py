import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.reduced_layers_utils import (
    ReducedLayersServerBase,
    assert_basic_generation_response,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=240, suite="stage-b-test-small-1-gpu")


class TestLlamaSanity(ReducedLayersServerBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

    def test_sanity(self):
        assert_basic_generation_response(self, self.base_url)


if __name__ == "__main__":
    unittest.main()
