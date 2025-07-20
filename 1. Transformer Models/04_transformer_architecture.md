# How do Transformers work?

This section touches on the architecture of Transformers, as well as a deep dive on some concepts like attention, encoder-decoder architecture, etc...

## A bit of Transformer history 

Some reference points of the history of Transformers : 

![History of transformers](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono-dark.svg)

## Transformers are language models

The Transformers in the image above have been trained as *laguage models*. That means they were trained with a lot of raw texts in a self-supervised fashion.

Self-supervised means there weren't human labels made on the data, meaning the model's objective is automatically computed from the inputs given.

When a model is trained this way, it develops a statistical understanding of the language is has been trained on, but is less useful for specific tasks. Because of this, the model goes through a process afterwards called *transfer learning* or *fine-tuning*. During this process, the fine-tuning is bade in a supervised way, using human-annotated labels, on a given task.

An example of one of these *fine-tuning* tasks is what's called as *causal language modelling*, which consists in predicting the next word in a sentence having read the *n* previuos words.

Here we can see a visual representation of this task:

![Causal Language Modelling](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/causal_modeling-dark.svg)

Another example of a task is *masked modelling*, where the model is expected to predict the masked word.

![Masked Modelling](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/masked_modeling-dark.svg)

## Transformers are big models

Apart from a couple of outliers, the strategy for better performance seems to be increasing the models' sizes as well as the data amount they are pretrained on.

![Model sizes](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/model_parameters.png)

Training models, as one would suspect, requires a lot of data. Not only is this costly, but it also has a strong environmental inpact :

![Carbon footprint](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/carbon_footprint-dark.svg)

If every student, research team or organization trained a model from scratch, the carbon footprint will be massive. 

This is why it's so important to share the trained weights and building on top of already trained weights. It reduces the overall compute cost and carbon footprint on the community!

## Transfer learning

Training a model from scratch has the model have randomly initialized weights, and the training starts without any prior knowledge.

