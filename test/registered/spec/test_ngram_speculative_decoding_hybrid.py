"""Regression test for NGRAM speculative decoding on hybrid-linear attention models.

Reproduces https://github.com/sgl-project/sglang/issues/20721 where
NgramVerifyInput was missing the `topk` attribute required by
HybridLinearAttnBackend during CUDA-graph capture.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_NGRAM_HYBRID,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="stage-b-test-large-1-gpu")


class TestNgramSpecDecodingHybridLinear(CustomTestCase):
    """Verify that NGRAM spec decoding starts and serves requests on a
    hybrid-linear model (Qwen3.5) without AttributeError on NgramVerifyInput."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_NGRAM_HYBRID
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--speculative-algorithm",
                "NGRAM",
                "--speculative-num-draft-tokens",
                "16",
                "--cuda-graph-max-bs",
                "4",
                "--mem-fraction-static",
                "0.8",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate_basic(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
