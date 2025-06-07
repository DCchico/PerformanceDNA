import torch
from transformers import BertTokenizer, BertForQuestionAnswering

def measure_latency(model, input_ids, token_type_ids, attention_mask, device, iterations=50):
    # Warm-up
    for _ in range(5):
        _ = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    # Timed
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    for _ in range(iterations):
        start_event.record()
        _ = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)
    return total_time / iterations

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    base_model = BertForQuestionAnswering.from_pretrained(model_name, return_dict=False).to(device).eval()

    # Create sample input (batch_size=8)
    context = "My name is Alice and I live in Wonderland."
    question = "Where do I live?"
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    batch_size = 8
    input_ids = inputs["input_ids"].expand(batch_size, -1).to(device)
    token_type_ids = inputs["token_type_ids"].expand(batch_size, -1).to(device)
    attention_mask = inputs["attention_mask"].expand(batch_size, -1).to(device)

    # Eager baseline
    eager_time = measure_latency(base_model, input_ids, token_type_ids, attention_mask, device)
    print(f"Eager average latency (ms): {eager_time:.2f}")

    # TorchScript Trace (assuming static shape)
    try:
        traced_model = torch.jit.trace(base_model, (input_ids, token_type_ids, attention_mask))
        traced_model.eval()
        traced_model.to(device)
        trace_time = measure_latency(traced_model, input_ids, token_type_ids, attention_mask, device)
        print(f"TorchScript Trace average latency (ms): {trace_time:.2f}")
    except Exception as e:
        print("Failed to trace:", e)

    # TorchScript Script
    try:
        scripted_model = torch.jit.script(base_model)
        scripted_model.eval()
        scripted_model.to(device)
        script_time = measure_latency(scripted_model, input_ids, token_type_ids, attention_mask, device)
        print(f"TorchScript Script average latency (ms): {script_time:.2f}")
    except Exception as e:
        print("Failed to script:", e)

if __name__ == "__main__":
    main()


