from vllm import LLM, SamplingParams
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument("--model",
                        type=str,
                        choices=[
                             "Mistral",
                             "TinyLlama",
                             "Phi-2",
                             "Gemma"
                         ])

    return parser.parse_args()


def Generate(llm):
    prompts = [
        "Translate 'hello, how are you?' into Korean:",
        "Briefly explain about relativity:",
        "Who is your favorite character from Three Kingdoms?",
        "Give me 10 remarkable stories about Liu Bei, emperor of Shu Han:",
        "What is MIG from nvidia?:",
        "What key algorithm is FFT based on?:",
        "Tell me what to do: to be or not to be",
        "Can you give me some tips to be a good runner?",
        "What is silicon?",
        "Rate this mantra; 'Pain is inevitable, suffering is an option'",
        "Suggest me a Lunch menu:",
        "What would be a meaning of human life?:",
        "What animals have longest longevity?:",
        "Teach me a recipe of Mak-geol-li:",
        "Summarize today's temperature:",
        "How much volume of water is perfect for cooking a soup?:",
        "Let me know how to make relationship with good people",
        "Suggest me good tourist sites of Korea:",
        "How many dogs are living in this planet?:",
        "How many tokens you can handle?:",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=150)
    # temperature: randomness(the higher, the more randomness)
    # top_p: threshold
    # top_k: top_k
    # max_tokens: max tokens to generate

    print("generating")
    try:
        outputs=llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"ERROR during gen: {e}")
        exit()
    print("DONE")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-"*20)
        print(f"prompt: {prompt!r}")
        print(f"generated: {generated_text!r}")
        print("-"*20)


if __name__ == '__main__':
    args = get_args()

    if args.model == "Mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif args.model == "TinyLlama":
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    elif args.model == "Phi-2":
        model_name = "miscrosoft/phi-2"
    elif args.model == "Gemma":
        model_name = "google/gemma-2b-it"

    # Initialize vllm engine
    try:
        llm = LLM(model=model_name)
    except Exception as e:
        print(f"LLM init error: {e}")
        exit()

    
