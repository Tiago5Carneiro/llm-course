# Tranformer Models

## How do Transformers work?

This section touches on the architecture of Transformers, as well as a deep dive on some concepts like attention, encoder-decoder architecture, etc...

### A bit of Transformer history 

Some reference points of the history of Transformers : 

![History of transformers](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono-dark.svg)

### Transformers are language models

The Transformers in the image above have been trained as *laguage models*. That means they were trained with a lot of raw texts in a self-supervised fashion.

Self-supervised means there weren't human labels made on the data, meaning the model's objective is automatically computed from the inputs given.

When a model is trained this way, it develops a statistical understanding of the language is has been trained on, but is less useful for specific tasks