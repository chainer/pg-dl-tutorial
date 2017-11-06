# 6. Add Extensions to the Trainer object

The [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) extensions provide the following capabilites:

- Save log files automatically ([LogReport](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.LogReport.html#chainer.training.extensions.LogReport))
- Display the training information to the terminal periodically ([PrintReport](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.PrintReport.html#chainer.training.extensions.PrintReport))
- Visualize the loss progress by plottig a graph periodically and save it as an image file ([PlotReport](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.PlotReport.html#chainer.training.extensions.PlotReport))
- Automatically serialize the state periodically ([snapshot()](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.snapshot.html#chainer.training.extensions.snapshot) / [snapshot_object()](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.snapshot_object.html#chainer.training.extensions.snapshot_object))
- Display a progress bar to the terminal to show the progress of training ([ProgressBar](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.ProgressBar.html#chainer.training.extensions.ProgressBar))
- Save the model architechture as a Graphviz’s dot file ([dump_graph](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.dump_graph.html#chainer.training.extensions.dump_graph))

To use these wide variety of tools for your tarining task, pass [Extension](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Extension.html#chainer.training.Extension) objects to the [extend()](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer.extend) method of your [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) object.

## [LogReport](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.LogReport.html#chainer.training.extensions.LogReport)

Collect `loss` and `accuracy` automatically every `'epoch'` or `'iteration'` and store the information under the `log` file in the directory specified by the `out` argument when you create a [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) object.

## [snapshot()](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.snapshot.html#chainer.training.extensions.snapshot)

The [snapshot()](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.snapshot.html#chainer.training.extensions.snapshot) method saves the [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) object at the designated timing (defaut: every epoch) in the directory specified by `out`. The [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) object, as mentioned before, has an [Updater](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Updater.html#chainer.training.Updater) which contains an [Optimizer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Optimizer.html#chainer.Optimizer) and a model inside. Therefore, as long as you have the snapshot file, you can use it to come back to the training or make inferences using the previously trained model later.

## [snapshot_object()](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.snapshot_object.html#chainer.training.extensions.snapshot_object)

By using this, you can save the particular object (for example, the model object wrapped by [Classifier](https://docs.chainer.org/en/latest/reference/generated/chainer.links.Classifier.html#chainer.links.Classifier)) as a separeted snapshot. [Classifier](https://docs.chainer.org/en/latest/reference/generated/chainer.links.Classifier.html#chainer.links.Classifier) is a [Chain](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain) object which keeps the model that is also a [Chain](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain) object as its `predictor` property, and all the parameters are under the `predictor`, so taking the snapshot of `predictor` is enough to keep all the trained parameters basically.

## [dump_graph()](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.dump_graph.html#chainer.training.extensions.dump_graph)

This method save the structure of the computational graph of the model. The graph is saved in the [Graphviz](http://www.graphviz.org/)'s `dot` format. The output location (directory) to save the graph is set by the `out` argument of [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer).

## [Evaluator](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.Evaluator.html#chainer.training.extensions.Evaluator)

The [Iterator](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.Iterator.html#chainer.dataset.Iterator) that uses the evaluation dataset and the model object are required to use [Evaluator](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.Evaluator.html#chainer.training.extensions.Evaluator). It evaluates the model using the given dataset (typically it’s a validation dataset) at the specified timing interval.

## [PrintReport](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.PrintReport.html#chainer.training.extensions.PrintReport)

It outputs the spcified values to the standard output.

## [PlotReport](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.PlotReport.html#chainer.training.extensions.PlotReport)

[PlotReport](https://docs.chainer.org/en/latest/reference/generated/chainer.training.extensions.PlotReport.html#chainer.training.extensions.PlotReport) plots the values specified by its arguments saves it as a image file which has the same naem as the `file_name` argument.

---

Each [Extension](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Extension.html#chainer.training.Extension) class has different options and some extensions are not mentioned here. And one of other important feature is, for instance, by using the [trigger](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Extension.html#chainer.training.Extension.trigger) option, you can set individual timings to fire the [Extension](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Extension.html#chainer.training.Extension).

**To know more details of all extensions, please take a look at the official document: [Trainer extensions](https://docs.chainer.org/en/stable/reference/extensions.html)**
