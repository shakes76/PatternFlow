from simple_parsing import ArgumentParser

from tk_train import StyleGanTrainer
from training_arguments import TrainingArguments

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest="training_arguments")
    args = parser.parse_args()
    print(args.training_arguments)

    trainer = StyleGanTrainer(args.training_arguments)
    trainer.fit(continue_from_previous_checkpoint=args.training_arguments.continue_from_previous_checkpoint,
                start_point=args.training_arguments.start_point)
