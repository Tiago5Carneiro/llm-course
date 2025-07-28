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

Training a model from scratch has the model have randomly initialized weights, and the training starts without any prior knowledge. This act is called *Pretraining*

![Pretraining](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining-dark.svg)

This pretraining is done with very large amounts of data, taking usually weeks to complete.

*Fine-tuning* on the other hand is the act of training the model **after** a model has been pretrained. This consists in grabbing the model after having been pretrained and train it to the specific task at hand.

#### Why not train the model for the final use right away?

There's a couple of reasons for that :

- The pretraining has already been done with data that is very similar to the fine-tuning task.
- Since the pretraining has already been done with a lot of data, the fine-tuning requires little data to get decent results.
- For these reasons, the amount of time and resources needed to get good results are much lower.

A good example is a model trained for the English language, that is then fine-tuned with arXiv corpus, which results in a science/research focused model. Thanks to the pretraining present, the cost of the final model is much lower, since all it's missing is fine-tuning the model.

![Benifit of fine-tuning](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/finetuning-dark.svg)

Here's 2 final things to keep in mind when thinking of fine-tuning :

- The lower time, data required, financial and environmental costs are very big reasons why we should fine-tune instead of training from scratch.
- This process earns better results than training from scratch, so if possible always fine-tune a model that is a close to the required task as possible.

## General Transformer Architecture

A model is composed primarily of 2 blocks :

- **Encoder** : It receives an input and builds a representation of it (it's features). That means the encoder models are optimized to acquire the understanding of it's input.
- **Decoder** : This block uses the understanding from the encoder (features) along with other inputs to generate a target sentence. It is therefor optimized for generating outputs.

![Encoder-Decoder](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks-dark.svg)

Each part can be used independently, depending on what's required :

- **Enconder-only models** : Good for tasks that required understanding like sentence classification and named entity recognition.
- **Decoder-only models** : Excelente for generative tasks like text generation.
- **Encoder-Decoder models** or **sequence-to-sequence models** : Good for generative tasks that require input like translations and summarization.

## Attention Layers

These layers pretty much tell the model which words to pay most attention to (and which ones to pretty much ignore). 

Here's an example. Let's say you want to use a model to translate from English the sentence "You like this course" to French. In order for the model to properly translate the word "like" it would have to pay attention to the word "You", since the conjugation matters in french. In that vein, it can pretty much ignore all other words from that sentence as those will not affect the translation of the word "like". As sentences get more complex (and have more complex grammar rules), the model would need to pay special attention to words that might appear farther away to properly translate each word.

This concept also applies to natural language. A word by itself has a meaning, but the meaning is deeply affected by it's context, which could be any word (or words) before or after said word.

## The original architecture

The Transformers architecture was originally designed for translation. During training, the encoder received inputs (sentences) in a certain language, while de decoder received the same sentences in the target language. The decoder works sequentially, which means it can only pay attention to words that have already been translated. For example, when trying to predict a forth word, it can only look at the 3 previous words and all the inputs from the encoder.

To speed up the training, the decoder was only allowed to use the words before the current word he's trying to predict.

The original architecture was the following :

![The original architecture](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers-dark.svg)

> ⚠️ **Note:** The first attention layer in a decoder pays attention to all (past) inputs of the decoder, but the second attention layer uses the output of the encoder. It can thus access the whole input sentence to best predict the current word. This is very important as different languages have different grammatical rules that put words in different orders, or seom context provided later in the sentence may be helpful to determine the best translation of a given word.

The *attention-mask* can be used in the encoder/decoder to prevent the model from paying attention to some special words like, for example, the special padding word used to make all inputs the same length when batching together senteces.

## Architecture vs. checkpoints

This course touches on a lot of concepts. There are 3 main ones that should not be confused on, *architectures*, *checkpoints* and *models* : 

- **Architecture** : This is the skeleton, the definition of each layer and operation that happens within the model.
- **Checkpoints** : These are the weights that will be loaded in a given architecture.
- **Model** : This term can be used for both *architectures* and *checkpoints*, however, when reducing ambiguity matters, both terms will be used in place of model.

> ℹ️ **Info:** A good example of this is BERT, which is an architecture while `bert-base-cased` is a set of weights trained by the Google team for the first release of BERT. Both "the BERT model" and "the `bert-base-cased` model" are valid descriptions of it.

⬅️ [Back to Transformer Pipelines](03_transformer_pipelines.md) ➡️ [Next: Transformer Task Execution](05_transformer_task_execution.md)