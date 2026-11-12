import argparse
import torch

from config import CONTEXT_LENGTH, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS
from models.model import MiniGPT
from tokenizer import Tokenizer

context_length = CONTEXT_LENGTH

max_new_tokens = 200
temperature = 0.7
top_k = 40
top_p = 0.9
min_p = 0.08
repetition_penalty = 1.15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer()

# Modellen bygges, når checkpointet er indlæst
model = None


def _detect_architecture(saved_cfg, state_dict):
    if "architecture" in saved_cfg:
        return saved_cfg["architecture"]
    if any("attention.qkv.weight" in k for k in state_dict):
        return "modern"
    return "legacy"


def _apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0 or not generated_ids:
        return logits
    logits = logits.clone()
    unique_ids = torch.tensor(list(set(generated_ids)), dtype=torch.long, device=logits.device)
    token_logits = logits[:, unique_ids]
    logits[:, unique_ids] = torch.where(
        token_logits < 0,
        token_logits * penalty,
        token_logits / penalty,
    )
    return logits


def _filter_logits(logits, top_k=0, top_p=1.0, min_p=0.0):
    if min_p is not None and 0.0 < min_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < (max_probs * min_p), float("-inf"))

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    return logits


@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    min_p=0.08,
    repetition_penalty=1.15,
    greedy=False,
):

    # Convert text to tokens; trim left if multi-turn history approaches context_length
    tokens = tokenizer.encode(prompt)
    max_prompt_tokens = max(8, context_length - 24)
    if len(tokens) > max_prompt_tokens:
        tokens = tokens[-max_prompt_tokens:]
    prompt_len = len(tokens)
    decoded_prompt = tokenizer.decode(tokens)

    # Convert to tensor: (1, T)
    x = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    # Generate one token at a time
    for _ in range(max_new_tokens):

        # Only use the last context_length tokens
        x_context = x[:, -context_length:]

        # Forward pass
        logits = model(x_context)

        # Select logits from the last position
        logits = logits[:, -1, :]

        generated_ids = x[0, prompt_len:].tolist()
        logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

        if greedy:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            if temperature <= 0:
                raise ValueError("temperature must be greater than zero")
            logits = logits / temperature
            logits = _filter_logits(logits, top_k=top_k, top_p=top_p, min_p=min_p)
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        # Append next token
        x = torch.cat((x, next_token), dim=1)

        # Decode generated text
        text = tokenizer.decode(x[0].tolist())

        # Only check newly generated text
        new_text = text[len(decoded_prompt):]

        # Stop at the next response boundary
        if "<SEP>" in new_text or "<END>" in new_text:
            break

    # Decode text
    text = tokenizer.decode(x[0].tolist())

    # Fjern prompt, så vi kun får modellens svar
    response = text[len(decoded_prompt):]

    # Fjern separatorer
    response = response.split("<SEP>")[0].split("<END>")[0]

    return response.strip()


def run_interactive_chat(args):
    history = []
    print("Interactive chat started! Commands: '/reset' to clear history, '/exit' or Ctrl+D to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Exiting chat.")
            break
        if user_input.lower() == "/reset":
            history.clear()
            print("[History cleared]\n")
            continue

        # Build multi-turn context window matching DialogDataset format:
        # q1<SEP>a1<END>q2<SEP>a2<END>...qN<SEP>
        selected_turns = []
        current_prompt = user_input + "<SEP>"
        budget = max(8, context_length - 32)
        used = len(tokenizer.encode(current_prompt))

        for prev_q, prev_a in reversed(history):
            turn_text = prev_q + "<SEP>" + prev_a + "<END>"
            turn_tokens = len(tokenizer.encode(turn_text))
            if used + turn_tokens > budget:
                break
            selected_turns.append(turn_text)
            used += turn_tokens

        full_prompt = "".join(reversed(selected_turns)) + current_prompt
        reply = generate(
            full_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            greedy=args.greedy,
        )
        print(f"Bot: {reply}\n")
        history.append((user_input, reply))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT")
    parser.add_argument("--checkpoint", default="miniGPT_full_best.pth")
    parser.add_argument("--prompt", default="How are you today?<SEP>")
    parser.add_argument("--chat", action="store_true", help="Run interactive multi-turn chat")
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--top-k", type=int, default=top_k)
    parser.add_argument("--top-p", type=float, default=top_p)
    parser.add_argument("--min-p", type=float, default=min_p)
    parser.add_argument("--repetition-penalty", type=float, default=repetition_penalty)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Brug modelstørrelsen fra checkpointet, så både nye og ældre checkpoints virker
    model_config = checkpoint.get("model_config", {})
    context_length = model_config.get("context_length", CONTEXT_LENGTH)
    tie_weights = model_config.get("tie_weights", "outputLayer.bias" not in state_dict)
    architecture = _detect_architecture(model_config, state_dict)
    vocab_size = model_config.get(
        "vocab_size", state_dict["tokens_embedding.weight"].shape[0]
    )

    model = MiniGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=model_config.get("embedding_dim", EMBEDDING_DIM),
        num_heads=model_config.get("num_heads", NUM_HEADS),
        num_layers=model_config.get("num_layers", NUM_LAYERS),
        tie_weights=tie_weights,
        architecture=architecture,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if args.chat:
        run_interactive_chat(args)
    else:
        # Samme format som i træningen: spørgsmål uden mellemrum før <SEP>
        prompt = args.prompt.split("<SEP>")[0].strip() + "<SEP>"

        response = generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            greedy=args.greedy,
        )

        print("\nGenerated text:")
        print(response)