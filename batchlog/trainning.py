def progress(trainer, mb, frequency=20, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(
                mb, training_loss, eval_error * 100))

    return mb, training_loss, eval_error
