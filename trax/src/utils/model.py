from trax.models import reformer
from trax import layers as tl


def NERModel(tags, vocab_size, d_model=50, predict=False):
    model = tl.Serial(
        reformer.Reformer(
            vocab_size,
            d_model,
            ff_activation=tl.LogSoftmax,
            mode="eval" if predict else "train",
        ),
        tl.Dense(tags),
        tl.LogSoftmax(),
    )
    return model
