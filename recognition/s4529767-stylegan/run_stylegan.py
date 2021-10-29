from tk_train import StyleGanTrainer

if __name__ == '__main__':
    #trainer = StyleGanTrainer("cpu")
    trainer = StyleGanTrainer('cuda:0')
    trainer.fit(continue_from_previous_checkpoint=True, start_point=300)
