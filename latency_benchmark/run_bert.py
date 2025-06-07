import warnings
warnings.filterwarnings("ignore")  # Hide non-critical warnings

import torch
from transformers import BertTokenizer, BertForQuestionAnswering

def measure_latency(model, device, input_ids, token_type_ids, attention_mask, timed_iterations=50):
    """
    Measures average forward latency (ms) using CUDA events.
    """
    # Warm-up
    warmup_iters = 5
    for _ in range(warmup_iters):
        _ = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time_ms = 0.0

    for _ in range(timed_iterations):
        start_event.record()
        _ = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        end_event.record()
        torch.cuda.synchronize()
        total_time_ms += start_event.elapsed_time(end_event)

    return total_time_ms / timed_iterations

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load a Hugging Face BERT QA model
    #    'bert-large-uncased-whole-word-masking-finetuned-squad' is ~1.3GB
    #    'bert-base-uncased' is smaller (but not specifically QA-finetuned)
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    base_model = BertForQuestionAnswering.from_pretrained(model_name).to(device)
    base_model.eval()

    # 2) Create an example question/context
    context = "My name is Alice and I live in Wonderland."
    question = "Where do I live?"

    # 3) Tokenize and replicate for a certain batch size
    batch_size = 8
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].expand(batch_size, -1).to(device)
    token_type_ids = inputs["token_type_ids"].expand(batch_size, -1).to(device)
    attention_mask = inputs["attention_mask"].expand(batch_size, -1).to(device)

    # 4) Define config scenarios
    scenarios = [
        ("Eager", None, None),
        ("aot_eager", "aot_eager", "default"),
        ("cudagraphs", "cudagraphs", "default"),
        ("torch_tensorrt", "torch_tensorrt", "default"),
        # Uncomment if you have Triton working on your system:
        # ("inductor", "inductor", "default"),
    ]

    timed_iterations = 100
    results = []

    for scenario_name, backend, mode in scenarios:
        print(f"\n--- Benchmarking {scenario_name} ---")
        if backend is None:
            # Plain Eager
            test_model = base_model
        else:
            try:
                test_model = torch.compile(base_model, backend=backend, mode=mode)
            except Exception as e:
                print(f"  [Skipping] {scenario_name} due to error:", e)
                continue

        latency_ms = measure_latency(
            test_model, device, input_ids, token_type_ids, attention_mask, timed_iterations
        )
        print(f"  Average latency: {latency_ms:.2f} ms, batch_size={batch_size}")
        results.append((scenario_name, latency_ms))

    print("\n=== Final Results ===")
    for name, lat_ms in results:
        print(f"{name}: {lat_ms:.2f} ms")

if __name__ == "__main__":
    main()
