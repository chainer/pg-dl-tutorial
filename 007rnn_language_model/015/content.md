### 2.2.11 Setup and run trainer

Let's make an `trainer` object and start the training! Note that we add an `eval_hook` to the `Evaluator` extension to reset the internal states before starting evaluation process. It can prevent to use training data during evaluating the model.
