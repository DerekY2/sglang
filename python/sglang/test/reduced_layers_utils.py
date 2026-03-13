import gc
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import requests
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import get_device, kill_process_tree
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

REDUCED_LAYER_COUNT = 2
REDUCED_LAYER_SEED = 1234


def make_reduced_layer_override_args(
    num_hidden_layers: int = REDUCED_LAYER_COUNT,
    extra_override_args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    override_args = {"num_hidden_layers": num_hidden_layers}
    if extra_override_args:
        override_args.update(extra_override_args)
    return override_args


def build_dummy_server_args(
    *,
    num_hidden_layers: int = REDUCED_LAYER_COUNT,
    extra_override_args: Optional[dict[str, Any]] = None,
    trust_remote_code: bool = False,
    extra_server_args: Optional[list[str]] = None,
) -> list[str]:
    server_args = [
        "--load-format",
        "dummy",
        "--json-model-override-args",
        json.dumps(
            make_reduced_layer_override_args(num_hidden_layers, extra_override_args)
        ),
    ]
    if trust_remote_code:
        server_args.append("--trust-remote-code")
    if extra_server_args:
        server_args.extend(extra_server_args)
    return server_args


class ReducedLayersServerBase(DefaultServerBase):
    num_hidden_layers = REDUCED_LAYER_COUNT
    extra_override_args: dict[str, Any] = {}
    extra_server_args: list[str] = []
    trust_remote_code = False

    @classmethod
    def setUpClass(cls):
        cls.other_args = build_dummy_server_args(
            num_hidden_layers=cls.num_hidden_layers,
            extra_override_args=cls.extra_override_args,
            trust_remote_code=cls.trust_remote_code,
            extra_server_args=cls.extra_server_args,
        )
        super().setUpClass()


def assert_basic_generation_response(
    test_case,
    base_url: str,
    *,
    prompt: str = "Hello world",
    max_new_tokens: int = 8,
):
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": [prompt, "Count to three."],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
        timeout=30,
    )
    test_case.assertEqual(response.status_code, 200, msg=response.text)

    payload = response.json()
    test_case.assertEqual(len(payload), 2)
    for item in payload:
        test_case.assertIn("text", item)
        test_case.assertIn("meta_info", item)
        test_case.assertIsInstance(item["text"], str)
        test_case.assertGreaterEqual(item["meta_info"]["completion_tokens"], 1)


def assert_mixed_context_generation_batch_response(test_case, base_url: str):
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": [
                "Summarize this sentence in one word: reduced layers keep coverage.",
                "Continue this sentence: Mixture-of-experts models route tokens",
            ],
            "sampling_params": [
                {"temperature": 0, "max_new_tokens": 0},
                {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            ],
        },
        timeout=30,
    )
    test_case.assertEqual(response.status_code, 200, msg=response.text)

    payload = response.json()
    test_case.assertEqual(len(payload), 2)
    test_case.assertEqual(payload[0]["meta_info"]["completion_tokens"], 0)
    test_case.assertGreaterEqual(payload[1]["meta_info"]["completion_tokens"], 1)


def run_reduced_layers_sanity_case(
    test_case,
    model: str,
    *,
    trust_remote_code: bool = False,
    extra_server_args: Optional[list[str]] = None,
    extra_override_args: Optional[dict[str, Any]] = None,
    prompt: str = "Hello world",
    mixed_batch: bool = False,
    base_url: str = DEFAULT_URL_FOR_TEST,
    timeout: float = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
):
    process = popen_launch_server(
        model,
        base_url,
        timeout=timeout,
        other_args=build_dummy_server_args(
            extra_override_args=extra_override_args,
            trust_remote_code=trust_remote_code,
            extra_server_args=extra_server_args,
        ),
    )
    try:
        if mixed_batch:
            assert_mixed_context_generation_batch_response(test_case, base_url)
        else:
            assert_basic_generation_response(test_case, base_url, prompt=prompt)
    finally:
        kill_process_tree(process.pid)


@contextmanager
def create_reduced_layer_hf_checkpoint(
    model_name: str,
    *,
    num_hidden_layers: int = REDUCED_LAYER_COUNT,
    trust_remote_code: bool = False,
    dtype: torch.dtype = torch.bfloat16,
):
    with tempfile.TemporaryDirectory(prefix="reduced_layers_") as tmpdir:
        model_dir = Path(tmpdir)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(model_dir)

        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        config.num_hidden_layers = num_hidden_layers
        config.torch_dtype = str(dtype).replace("torch.", "")

        old_dtype = torch.get_default_dtype()
        torch.manual_seed(REDUCED_LAYER_SEED)
        torch.set_default_dtype(dtype)
        try:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=trust_remote_code
            )
        finally:
            torch.set_default_dtype(old_dtype)

        model.eval()
        model.save_pretrained(
            model_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
        del model
        del tokenizer
        gc.collect()
        yield str(model_dir)


def assert_reduced_layer_full_vocab_logprobs_close(
    test_case,
    model_path: str,
    *,
    prompt: str,
    trust_remote_code: bool = False,
    atol: float = 5e-3,
    rtol: float = 5e-3,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
    ).to(get_device())
    hf_model.eval()

    with torch.inference_mode():
        hf_logits = hf_model(input_ids.to(hf_model.device)).logits[0, -1].float()
    hf_logprobs = F.log_softmax(hf_logits, dim=-1).cpu()

    del hf_model
    del hf_logits
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    engine = Engine(
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        random_seed=REDUCED_LAYER_SEED,
        mem_fraction_static=0.55,
    )
    try:
        response = engine.generate(
            input_ids=input_ids[0].tolist(),
            sampling_params={"temperature": 1.0, "max_new_tokens": 1},
            return_logprob=True,
            token_ids_logprob=list(range(hf_logprobs.numel())),
        )
    finally:
        engine.shutdown()

    if isinstance(response, list):
        response = response[0]

    output_token_ids_logprobs = response["meta_info"]["output_token_ids_logprobs"][0]
    sglang_logprobs = torch.tensor(
        [value for value, _, _ in output_token_ids_logprobs], dtype=torch.float32
    )

    test_case.assertEqual(sglang_logprobs.numel(), hf_logprobs.numel())
    torch.testing.assert_close(sglang_logprobs, hf_logprobs, atol=atol, rtol=rtol)
