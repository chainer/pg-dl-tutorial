### 2.2.10 Setup optimizer

Prepare an optimizer. Here, we use `GradientClipping` to prevent gradient explosion. It automatically clip the gradient to be used to update the parameters in the model with given constant `gradclip`.
