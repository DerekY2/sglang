import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.reduced_layers_utils import (
    ReducedLayersServerBase,
    assert_basic_generation_response,
)
from sglang.test.test_utils import DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=180, suite="stage-b-test-large-1-gpu")


class TestQwen3MoESanity(ReducedLayersServerBase):
    model = DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST
    trust_remote_code = True
    extra_server_args = ["--mem-fraction-static", "0.7"]

    def test_sanity(self):
        assert_basic_generation_response(
            self,
            self.base_url,
            prompt="Explain MoE in one sentence.",
        )


if __name__ == "__main__":
    unittest.main()
