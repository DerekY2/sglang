import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.reduced_layers_utils import run_reduced_layers_sanity_case
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=300, suite="nightly-2-gpu", nightly=True)


class TestReducedLayersNightly2GPUSanity(CustomTestCase):
    CONFIGS = {
        "qwen3_moe_ep2": {
            "model": DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
            "trust_remote_code": True,
            "extra_server_args": ["--ep-size", "2", "--mem-fraction-static", "0.7"],
        },
    }

    def _run_case(self, cfg):
        run_reduced_layers_sanity_case(
            self,
            cfg["model"],
            trust_remote_code=cfg.get("trust_remote_code", False),
            extra_server_args=cfg.get("extra_server_args"),
        )


for _name, _cfg in TestReducedLayersNightly2GPUSanity.CONFIGS.items():
    setattr(
        TestReducedLayersNightly2GPUSanity,
        f"test_{_name}",
        lambda self, cfg=_cfg: self._run_case(cfg),
    )


if __name__ == "__main__":
    unittest.main()
