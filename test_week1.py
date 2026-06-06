"""
Weekend 1 — Exercise Tests

Importable Python script. Run directly:
    python tests/test_week1.py

Or import individual tests:
    from tests.test_week1 import test_lora_parameter_savings, run_all_tests
"""

import asyncio
import json
import os

import numpy as np
import tinker
from tinker_cookbook.renderers import get_renderer, get_text_content, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tqdm import tqdm

TINKER_API_KEY = os.environ["TINKER_API_KEY"]

def _find_jsonl(name="gsm8k_capitalised.jsonl"):
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in [here, os.path.join(here, ".."), os.getcwd()]:
        p = os.path.join(candidate, name)
        if os.path.exists(p):
            return os.path.normpath(p)
    return os.path.join(here, name)  # fallback — will error with a clear message

JSONL_PATH = _find_jsonl()


# ── Reference implementations ─────────────────────────────────────────────────

def create_connection(api_key: str, base_model: str, rank: int, renderer: str):
    service_client  = tinker.ServiceClient(api_key=api_key)
    training_client = service_client.create_lora_training_client(base_model=base_model, rank=rank)
    tokenizer       = training_client.get_tokenizer()
    renderer_obj    = get_renderer(renderer, tokenizer)
    return service_client, training_client, renderer_obj


def load_jsonl_as_datums(lines: list, renderer, max_length=None, last_message: bool = True) -> list:
    datums = []
    for line in tqdm(lines, desc="Converting JSONL to Datums"):
        row  = json.loads(line.strip())
        conv = row["messages"]
        mode = TrainOnWhat.LAST_ASSISTANT_MESSAGE if last_message else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        datums.append(conversation_to_datum(conv, renderer, max_length, mode))
    return datums


async def run_one_training_step(training_client, datums: list, learning_rate: float = 1e-4) -> float:
    fwdbwd_future = await training_client.forward_backward_async(datums, "cross_entropy")
    optim_future  = await training_client.optim_step_async(tinker.AdamParams(learning_rate=learning_rate))
    fwdbwd_result = await fwdbwd_future.result_async()
    await optim_future.result_async()
    logprobs = np.concatenate([out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs])
    weights  = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in datums])
    return float(-np.dot(logprobs, weights) / weights.sum())


async def run_training(training_steps, training_client, train_datums, learning_rate, batch_size):
    losses = []
    for step in tqdm(range(training_steps), desc="Training", unit="step"):
        batch = [train_datums[(step * batch_size + i) % len(train_datums)] for i in range(batch_size)]
        loss  = await run_one_training_step(training_client, batch, learning_rate)
        losses.append(loss)
    return losses


async def run_training_step(training_client, datums: list, learning_rate: float = 1e-4) -> float:
    fwdbwd_future = await training_client.forward_backward_async(datums, "cross_entropy")
    optim_future  = await training_client.optim_step_async(tinker.AdamParams(learning_rate=learning_rate))
    fwdbwd_result = await fwdbwd_future.result_async()
    await optim_future.result_async()
    logprobs = np.concatenate([out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs])
    weights  = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in datums])
    return float(-np.dot(logprobs, weights) / weights.sum())


async def ask(sampler, renderer, question: str, num_samples: int = 1,
              max_tokens: int = 512, temperature: float = 1.0) -> str:
    stop   = renderer.get_stop_sequences()
    inp    = renderer.build_generation_prompt([{"role": "user", "content": question}])
    result = await sampler.sample_async(
        inp, num_samples=num_samples,
        sampling_params=tinker.SamplingParams(max_tokens=max_tokens, stop=stop, temperature=temperature),
    )
    response, _ = renderer.parse_response(result.sequences[0].tokens)
    return get_text_content(response)


async def generate_from_tinker(sampling_client, prompts: list, renderer, max_tokens: int = 200) -> list:
    responses = []
    stop = renderer.get_stop_sequences()
    for prompt in prompts:
        inp    = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
        result = await sampling_client.sample_async(
            inp, num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=max_tokens, stop=stop),
        )
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        responses.append(get_text_content(response))
    return responses


async def run_pipeline(api_key, base_model="Qwen/Qwen3.5-4B", rank=32,
                       renderer="qwen3_disable_thinking", dataset_path=None,
                       n_train_steps=1, learning_rate=1e-4, batch_size=1,
                       max_tokens=512, eval_prompts=None):
    """
    Run the full fine-tuning and evaluation pipeline.

    Args:
        api_key:        Tinker API key
        base_model:     HuggingFace model ID to fine-tune
        rank:           LoRA adapter rank
        renderer:       renderer name for the model's chat template
        dataset_path:   path to the JSONL training file
        n_train_steps:  number of gradient steps
        learning_rate:  Adam learning rate
        batch_size:     training batch size
        max_tokens:     max tokens to generate per response
        eval_prompts:   list of prompts to evaluate; uses defaults if None

    Returns:
        (ft_responses, base_responses, losses, ft_sampler, base_sampler, training_client, renderer_client)
    """
    if dataset_path is None:
        dataset_path = JSONL_PATH
    if eval_prompts is None:
        eval_prompts = ["What is 2 + 2?"]
    with open(dataset_path) as f:
        lines = f.readlines()
    service_client, training_client, renderer_client = create_connection(api_key, base_model, rank, renderer)
    train_datums = load_jsonl_as_datums(lines, renderer_client, max_length=512)
    losses = await run_training(
        training_steps=n_train_steps, training_client=training_client,
        train_datums=train_datums, learning_rate=learning_rate, batch_size=batch_size,
    )
    ft_sampler    = await training_client.save_weights_and_get_sampling_client_async()
    base_training = service_client.create_lora_training_client(base_model=base_model, rank=1)
    base_sampler  = await base_training.save_weights_and_get_sampling_client_async()
    ft_responses   = [await ask(ft_sampler,   renderer_client, q, max_tokens=max_tokens) for q in eval_prompts]
    base_responses = [await ask(base_sampler, renderer_client, q, max_tokens=max_tokens) for q in eval_prompts]
    return ft_responses, base_responses, losses, ft_sampler, base_sampler, training_client, renderer_client


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_lora_parameter_savings(
    full_params: int,
    lora_a_params: int,
    lora_b_params: int,
    lora_params: int,
    pct_saved: float,
) -> None:
    """
    Check student answers for the LoRA parameter savings exercise.

    Expected values for a 3000×100 weight matrix with rank=8 LoRA:
        full_params   = 300,000
        lora_a_params = 800        (A: rank × d_in  = 8 × 100)
        lora_b_params = 24,000     (B: d_out × rank = 3000 × 8)
        lora_params   = 24,800
        pct_saved     ≈ 91.73%
    """
    assert full_params == 300_000, (
        f"full_params: expected 300,000 got {full_params:,}\n"
        "Hint: full_params = d_out × d_in = 3000 × 100"
    )
    assert lora_a_params == 800, (
        f"lora_a_params: expected 800 got {lora_a_params:,}\n"
        "Hint: A has shape (rank, d_in) = (8, 100)"
    )
    assert lora_b_params == 24_000, (
        f"lora_b_params: expected 24,000 got {lora_b_params:,}\n"
        "Hint: B has shape (d_out, rank) = (3000, 8)"
    )
    assert lora_params == 24_800, (
        f"lora_params: expected 24,800 got {lora_params:,}\n"
        "Hint: lora_params = lora_a_params + lora_b_params"
    )
    assert abs(pct_saved - 91.73) < 0.1, (
        f"pct_saved: expected ~91.73% got {pct_saved:.2f}%\n"
        "Hint: pct_saved = (1 - lora_params / full_params) × 100"
    )
    print(f"Full fine-tuning : {full_params:>10,} parameters")
    print(f"LoRA (rank=8)    : {lora_params:>10,} parameters  ({100 - pct_saved:.2f}% of full)")
    print(f"Parameters saved : {pct_saved:.2f}%")
    print("✓ test_lora_parameter_savings")


async def test_create_connection():
    svc, tc, rend = create_connection(TINKER_API_KEY, "Qwen/Qwen3.5-4B", 1, "qwen3_disable_thinking")
    assert hasattr(tc,   "forward_backward_async"),  "training_client missing forward_backward_async"
    assert hasattr(rend, "build_generation_prompt"), "renderer missing build_generation_prompt"
    print("✓ test_create_connection")
    return svc, tc, rend


def test_load_jsonl_as_datums(rend):
    with open(JSONL_PATH) as f:
        lines = f.readlines()[:3]
    datums = load_jsonl_as_datums(lines, rend, max_length=128)
    assert isinstance(datums, list), "should return a list"
    assert len(datums) == 3,         "one datum per line"
    assert all(hasattr(d, "loss_fn_inputs") for d in datums), "each datum needs loss_fn_inputs"
    assert all("weights" in d.loss_fn_inputs for d in datums), "each datum needs weights key"
    datums_all = load_jsonl_as_datums(lines, rend, max_length=128, last_message=False)
    assert len(datums_all) == 3, "one datum per line when last_message=False"
    print("✓ test_load_jsonl_as_datums")
    return datums


async def test_run_one_training_step(tc, datum):
    loss = await run_one_training_step(tc, [datum], learning_rate=1e-4)
    assert isinstance(loss, float) and loss > 0, "should return a positive float"
    print(f"✓ test_run_one_training_step  (loss={loss:.4f})")


async def test_run_training_step(tc, datum):
    loss = await run_training_step(tc, [datum], learning_rate=1e-4)
    assert isinstance(loss, float) and loss > 0, "should return a positive float"
    print(f"✓ test_run_training_step      (loss={loss:.4f})")


async def test_ask(sampler, rend):
    resp = await ask(sampler, rend, "What is 2 + 2?", max_tokens=20)
    assert isinstance(resp, str) and len(resp) > 0, "should return a non-empty string"
    print(f"✓ test_ask                    ('{resp[:50]}')")


async def test_generate_from_tinker(sampler, rend):
    resps = await generate_from_tinker(sampler, ["What is 2 + 2?"], rend, max_tokens=20)
    assert isinstance(resps, list) and len(resps) == 1, "should return a list with one item"
    assert isinstance(resps[0], str) and len(resps[0]) > 0, "response should be a non-empty string"
    print(f"✓ test_generate_from_tinker   ('{resps[0][:50]}')")


async def test_run_pipeline():
    ft, base, losses, ft_samp, base_samp, tc, rend = await run_pipeline(
        api_key=TINKER_API_KEY, rank=1, n_train_steps=1,
        eval_prompts=["What is 2 + 2?"],
    )
    assert len(ft) == len(base) == len(losses) == 1
    assert isinstance(ft[0], str)   and len(ft[0]) > 0
    assert isinstance(base[0], str) and len(base[0]) > 0
    print(f"✓ test_run_pipeline           (ft='{ft[0][:30]}...')")


async def run_all_tests():
    print("Running Weekend 1 tests...\n")

    test_lora_parameter_savings(
        full_params=300_000,
        lora_a_params=800,
        lora_b_params=24_000,
        lora_params=24_800,
        pct_saved=91.73,
    )

    svc, tc, rend = await test_create_connection()
    datums = test_load_jsonl_as_datums(rend)
    datum  = datums[0]

    await test_run_one_training_step(tc, datum)
    await test_run_training_step(tc, datum)

    sampler = await tc.save_weights_and_get_sampling_client_async()
    await test_ask(sampler, rend)
    await test_generate_from_tinker(sampler, rend)
    await test_run_pipeline()

    print("\nAll tests passed ✓")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
