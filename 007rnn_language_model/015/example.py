updater = BPTTUpdater(train_iter, optimizer, bproplen, gpu)
trainer = training.Trainer(updater, (epoch, 'epoch'), out='ptb_result')

eval_model = model.copy()  # Model with shared params and distinct states
eval_rnn = eval_model.predictor
trainer.extend(extensions.Evaluator(
    val_iter, eval_model, device=gpu,
    # Reset the RNN state at the beginning of each evaluation
    eval_hook=lambda _: eval_rnn.reset_state()))

trainer.extend(extensions.LogReport(postprocess=compute_perplexity, trigger=(1, 'epoch')))
trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'perplexity', 'val_perplexity']), trigger=(1, 'epoch'))
trainer.extend(extensions.snapshot())
trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}'))

trainer.run()
