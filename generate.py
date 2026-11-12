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
repetition_penalty = 1.15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer()

# Modellen bygges, når checkpointet er indlæst
model = None


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


def _filter_logits(logits, top_k=0, top_p=1.0):
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
    repetition_penalty=1.15,
    greedy=False,
):

    # Convert text to tokens
    tokens = tokenizer.encode(prompt)
    prompt_len = len(tokens)

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
            logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        # Append next token
        x = torch.cat((x, next_token), dim=1)

        # Decode generated text
        text = tokenizer.decode(x[0].tolist())

        # Only check newly generated text
        new_text = text[len(prompt):]

        # Stop at the next response boundary
        if "<SEP>" in new_text or "<END>" in new_text:
            break

    # Decode text
    text = tokenizer.decode(x[0].tolist())

    # Fjern prompt, så vi kun får modellens svar
    response = text[len(prompt):]

    # Fjern separatorer
    response = response.split("<SEP>")[0].split("<END>")[0]

    return response.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT")
    parser.add_argument("--checkpoint", default="miniGPT_full_best.pth")
    parser.add_argument("--prompt", default="How are you today?<SEP>")
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--top-k", type=int, default=top_k)
    parser.add_argument("--top-p", type=float, default=top_p)
    parser.add_argument("--repetition-penalty", type=float, default=repetition_penalty)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Brug modelstørrelsen fra checkpointet, så ældre checkpoints stadig virker
    model_config = checkpoint.get("model_config", {})
    context_length = model_config.get("context_length", CONTEXT_LENGTH)
    tie_weights = model_config.get("tie_weights", "outputLayer.bias" not in state_dict)
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
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Samme format som i træningen: spørgsmål uden mellemrum før <SEP>
    prompt = args.prompt.split("<SEP>")[0].strip() + "<SEP>"

    response = generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy
    )
    

    print("\nGenerated text:")
    print(response)