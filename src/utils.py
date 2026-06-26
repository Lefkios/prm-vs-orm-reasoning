import mlx.core as mx
from mlx_lm import load, generate

def test_inference():
    print("Loading model...")
    model, tokenizer = load("models/qwen2.5-7b")
    
    prompt = """Solve this step by step:

Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning 
and bakes muffins for her friends every day with 4. She sells the remainder 
at the farmers' market daily for $2 per fresh duck egg. How much does she 
make every day at the farmers' market?

Let's think step by step:"""

    print("Generating response...\n")
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=500,
        verbose=True
    )
    
    print("\n--- Full Response ---")
    print(response)

if __name__ == "__main__":
    test_inference()