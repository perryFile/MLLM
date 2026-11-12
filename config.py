 # Modelstørrelse. Bruges af train.py og gemmes i hvert checkpoint,
# så generate.py kan bygge den samme model igen.

# Multi-turn samtalevindue
CONTEXT_LENGTH = 128

# EMBEDDING_DIM skal kunne deles med NUM_HEADS (og head_dim skal være lige til RoPE)
EMBEDDING_DIM = 192
NUM_HEADS = 4
NUM_LAYERS = 4
VOCAB_SIZE = 4000

# Dropout modvirker, at modellen lærer træningsdata udenad.
# Bruges under træning i pretrain og full mode.
DROPOUT = 0.25

# Andel af pretraining-data (DailyDialog), der blandes ind under full fine-tuning
# for at undgå catastrophic forgetting.
REPLAY_RATIO = 0.15

