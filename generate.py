import argparse
import torch

from config import CONTEXT_LENGTH, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS
from models.model import MiniGPT
from tokenizer import Tokenizer

context_length = CONTEXT_LENGTH

max_new_tokens = 200
temperature = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer()

# Modellen bygges, når checkpointet er indlæst
model = None

@torch.no_grad()
def generate(prompt, max_new_tokens=200, temperature=0.8, greedy=False):

    # Convert text to tokens
    tokens = tokenizer.encode(prompt)

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

        if greedy:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            if temperature <= 0:
                raise ValueError("temperature must be greater than zero")
            logits = logits / temperature
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
    parser.add_argument("--checkpoint", default="miniGPT_overfit_best.pth")
    parser.add_argument("--prompt", default="How are you today?<SEP>")
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Brug modelstørrelsen fra checkpointet, så ældre checkpoints stadig virker
    model_config = checkpoint.get("model_config", {})
    context_length = model_config.get("context_length", CONTEXT_LENGTH)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        embedding_dim=model_config.get("embedding_dim", EMBEDDING_DIM),
        num_heads=model_config.get("num_heads", NUM_HEADS),
        num_layers=model_config.get("num_layers", NUM_LAYERS)
    )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()

    # Samme format som i træningen: spørgsmål uden mellemrum før <SEP>
    prompt = args.prompt.split("<SEP>")[0].strip() + "<SEP>"

    response = generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        greedy=args.greedy
    )
    

    print("\nGenerated text:")
    print(response)