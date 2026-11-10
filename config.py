 # Modelstørrelse. Bruges af train.py og gemmes i hvert checkpoint,
# så generate.py kan bygge den samme model igen.

# Længste eksempel i datasættet passer inden for 64 tokens
CONTEXT_LENGTH = 64

# EMBEDDING_DIM skal kunne deles med NUM_HEADS
EMBEDDING_DIM = 192
NUM_HEADS = 4
NUM_LAYERS = 4
VOCAB_SIZE = 4000

# Dropout modvirker, at modellen lærer træningsdata udenad.
# Bruges under træning i pretrain og full mode.
DROPOUT = 0.25

