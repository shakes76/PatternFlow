from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """
    Arguments for training the StyleGan network.
    """
    device: Optional[str] = field(
        default="cuda:0",
        metadata={"help": "The device training is executed on."}
    )
    continue_from_previous_checkpoint: Optional[str] = field(
        default=False,
        metadata={"help": "Load existing model from the checkpoint and continue training from there."}
    )
    start_point: Optional[str] = field(
        default=0,
        metadata={"help": "If continue_from_previous_checkpoint is set to True, continue training from the checkpoint "
                          "start_point"}
    )
    start_progressive_level: Optional[int] = field(
        default=1,
        metadata={
            "help": "Start level for progressive training."
        },
    )
    max_progressive_level: Optional[int] = field(
        default=7,
        metadata={
            "help": "Maximum progressive level for progressive training. Depends on image resolution."
        },
    )

