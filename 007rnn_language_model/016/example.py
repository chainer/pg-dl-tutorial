eval_rnn.reset_state()
evaluator = extensions.Evaluator(test_iter, eval_model, device=gpu)
result = evaluator()
print('test perplexity:', np.exp(float(result['main/loss'])))
