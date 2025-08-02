# How 🤗 Transformers solve tasks

## Transformer models for language 

Language models are at the heart of modern NLP. They’re designed to understand and generate human language by learning the statistical patterns and relationships between words or tokens in text.

The Transformer was initially designed for machine translation, and since then, it has become the default architecture for solving all AI tasks. Some tasks lend themselves to the Transformer’s encoder structure, while others are better suited for the decoder. Still, other tasks make use of both the Transformer’s encoder-decoder structure.

## How language models work

Language models are trained to predict the next word given a context of surrounding words. This makes them have a foundational knowledge that translates well for other tasks.

There are 2 main approaches for training a model :

- **Masked language modeling (MLM)** : This is used by encoder models like BERT, in this approach words in a sentence are randomly masked and the model has to try to predict them from the surrounding context (looking at words both before and after the masked one).
- **Casual language modeling (CLM)** : This approach is used by decoder models like GPT, where the model tries to predict the next word from the context previously provided. The model can only have the context to the left of the word (previous tokens) in order to predict the next one.

## Type of language models

In the Transformers library, language models generally fall into three architectural categories:

- **Encoder-only models** (like BERT): These models use a bidirectional approach to understand context from both directions. They’re best suited for tasks that require deep understanding of text, such as classification, named entity recognition, and question answering.
- **Decoder-only models** (like GPT, Llama): These models process text from left to right and are particularly good at text generation tasks. They can complete sentences, write essays, or even generate code based on a prompt.
- **Encoder-decoder models** (like T5, BART): These models combine both approaches, using an encoder to understand the input and a decoder to generate output. They excel at sequence-to-sequence tasks like translation, summarization, and question answering.