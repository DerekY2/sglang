import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.reduced_layers_utils import (
    assert_reduced_layer_full_vocab_logprobs_close,
    create_reduced_layer_hf_checkpoint,
    run_reduced_layers_sanity_case,
)
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
    CustomTestCase,
)

register_cuda_ci(est_time=600, suite="nightly-1-gpu", nightly=True)


class TestReducedLayersNightly1GPUSanity(CustomTestCase):
    CONFIGS = {
        "qwen3_moe_fp4_precision": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
            "trust_remote_code": True,
            "extra_server_args": [
                "--quantization",
                "modelopt_fp4",
                "--mem-fraction-static",
                "0.7",
            ],
        },
    }

    def _run_case(self, cfg):
        run_reduced_layers_sanity_case(
            self,
            cfg["model"],
            trust_remote_code=cfg.get("trust_remote_code", False),
            extra_server_args=cfg.get("extra_server_args"),
            mixed_batch=cfg.get("mixed_batch", False),
        )


for _name, _cfg in TestReducedLayersNightly1GPUSanity.CONFIGS.items():
    setattr(
        TestReducedLayersNightly1GPUSanity,
        f"test_{_name}",
        lambda self, cfg=_cfg: self._run_case(cfg),
    )


class TestQwen3MoEAllcloseToHF(CustomTestCase):
    def test_allclose_to_hf(self):
        with create_reduced_layer_hf_checkpoint(
            DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
            trust_remote_code=True,
        ) as model_path:
            assert_reduced_layer_full_vocab_logprobs_close(
                self,
                model_path,
                prompt="MoE routing works by",
                trust_remote_code=True,
            )


if __name__ == "__main__":
    unittest.main()
