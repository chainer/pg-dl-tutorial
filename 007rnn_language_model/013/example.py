rnn = RNNLM(n_vocab, unit)
model = L.Classifier(rnn)
model.compute_accuracy = False  # we only want the perplexity
