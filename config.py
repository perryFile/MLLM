# Modelstørrelse. Bruges af train.py og gemmes i hvert checkpoint,
# så generate.py kan bygge den samme model igen.

# Længste eksempel i datasættet er 63 tokens
CONTEXT_LENGTH = 64

# EMBEDDING_DIM skal kunne deles med NUM_HEADS
EMBEDDING_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6

# Dropout modvirker, at modellen lærer træningsdata udenad.
# Bruges kun under træning i full mode.
<<<<<<< HEAD
DROPOUT = 0.2
=======
DROPOUT = 0.1
>>>>>>> 3ad93a8e641a603edebf4dbf0836bde6f4a6942a

