# Transformers, what can they do?

Firstly, what even is a transformer?

**Transformer** is a neural network arquitecture invented by Google in 2017. It's a core component in many modern LLMs.

HuggingFace's opensource library contains a Transformers library, which I will be using.

Let's start with the `pipeline()` function, which returns an object that represents a pipeline. 

This pipeline is the connection of a model with it's necessary preprocessing and postprocessing steps, which allows us to input text and get an intelligible answer. 

It is the most high-level API, as all the necessary steps to transform the input into readable data for the model, the postprocessing necessary for human comprehension and the actual model computation are performed with it.

There are plenty of available pipelines :

### Text pipelines

- `text-generation`: Generate text from a prompt
- `text-classification`: Classify text into predefined categories
- `summarization`: Create a shorter version of a text while preserving key information
- `translation`: Translate text from one language to another
- `zero-shot-classification`: Classify text without prior training on specific labels
- `feature-extraction`: Extract vector representations of text

### Image pipelines

- `image-to-text`: Generate text descriptions of images
- `image-classification`: Identify objects in an image
- `object-detection`: Locate and identify objects in images

### Audio pipelines

- `automatic-speech-recognition`: Convert speech to text
- `audio-classification`: Classify audio into categories
- `text-to-speech`: Convert text to spoken audio

### Multimodal pipelines

- `image-text-to-text`: Respond to an image based on a text prompt

We will analyse how some of these pipelines work, what their

## Sentiment Analysis 

This is one of the available pipeline available in the transformers. It returns wether a sentence is 'POSITIVE' or 'NEGATIVE', with the associated confidence percentage.

### First exercise

For the [first exercise](transformer_exercices.py#10), I ran a sentiment-analysis on the sentence : 

```python
"I'm so excited for the upcoming weekend"
```

Which gave the following result :

```python
[{'label': 'POSITIVE', 'score': 0.9996603727340698}]
```

Multiple sentence is also available, as we can see with [exercise 2](transformer_exercices.py#15). These sentences :

```python
["I am not happy with the service I received.", "That meal was delicious!"]
```

Generated the following output :

```python
[{'label': 'NEGATIVE', 'score': 0.9994329810142517}, {'label': 'POSITIVE', 'score': 0.9998834133148193}]
```

## Zero-Shot Classification

Another one of the available pipelines is the zero-shot classification. In this pipeline, we provide a text, as well as a list of candidate labs that we want the model to associate with.

### Exercise 3

With the following input :

```python
(
    "Portugal joined nato in 1949.",
    candidate_labels=["education", "politics", "buisiness"]
)
```

I got the following result : 

```python
{
    'sequence': 'Portugal joined nato in 1949.', 
    'labels': ['buisiness', 'politics', 'education'], 
    'scores': [0.5552445650100708, 0.2405625730752945, 0.05916651338338852]
}
```

This pipeline gets it's name from the fact that it isn't needed to fine-tune the model on the data that is being provided. It returns the probability scores for any list of labels.

## Text generation

For text generation, the main idea is that, provided with an initial prompt, the model auto-completes it by generating the remaining text.

With this pipeline we can select a maximum number of tokens used with `max_length` and a number of sequences with `num_return_sequences`.

> ⚠️ **Note:** At some point the functionality of `max_length` went from defining the maximum number of words to maximum number of tokens, however the course has not been updated with that in mind.

### Exercise 4

Given the following prompt :

```python
classifier(
        "The future of AI is",
        max_length=20,
        num_return_sequences=1
)
```

The pipeline provided me with :

```python
[{'generated_text': "The future of AI is unclear. The most likely scenario is that an AI is able to predict the future behavior of its fellow human beings, and may even understand how to avoid the consequences of those decisions.\n\nIn the past, we have seen robots making the same choices or making the same decisions over and over again. For example, in the 1970s, some researchers suggested that humans could choose between saving our lives and helping our relatives and friends. But in the 1980s, scientists began to see how machines could do that—and we began to see the effects of robot choices on human outcomes.\n\nWe are starting to see more and more of this. We're living in a world where we can't choose how we want our food or the clothes we wear. In recent years, some researchers have proposed that human beings may be able to do things that humans can't do, including:\n\nStop having children\n\nStop going to the doctor for help\n\nHelp people with chronic diseases\n\nStop breastfeeding\n\nCreate and use robots for medical research\n\nIn this sense, these proposals are pretty much on track. The work on artificial intelligence has been moving in this direction for a while now, and it's only now that machines are starting to take a more human-centered"}]
```

## Using any model from the Hub in a pipeline

The previous exercises were done without defining a specific model for the pipelines. However, a specific model can be defined.

We can find more models to select and use [in this page](https://huggingface.co/models). These models can be tested using inference providers, which we can find [here](https://huggingface.co/docs/inference-providers/en/index)

### Exercise 5

Defining the generator as :

```python
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
```

And running : 
```python
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

We get :

```python
[{'generated_text': 'In this course, we will teach you how to design and build three-dimensional objects using the program. The object you will create will be a simple one, but you will learn how to build it using the program. You will also learn how to use the object to create other objects.\n\n\nThe basic parts of the program are the following:\n\n  1. The object. It will be a cube, square, circle, or any other shape.\n  2. The program will be a simple program that will display the object. The program will have a button and the object will be on the button.\n  3. The program will also have a function that will calculate the size of the object. The function will take in a number and return a number.\n  4. The function will also have a button that will display the object and the size of the object.\n  5. The object will also have a function that will display the size of the object. The function will take in a number and return a number.\n\n\nIf you want to learn how to design and build a cube, you can start by learning how to draw a cube on a piece of paper. Then, you can move on to building the cube using the program.\n\nThe program will allow you to create a cube using the program.'}, {'generated_text': 'In this course, we will teach you how to write a strong thesis statement, which is the most crucial part of your essay. Once you have a strong thesis statement, you can then start writing your essay. It will help you to organize your ideas and get your point across.\n\nIn this course, we will teach you how to write a strong thesis statement, which is the most crucial part of your essay. Once you have a strong thesis statement, you can then start writing your essay. It will help you to organize your ideas and get your point across.\n\nOur Thesis Statement Writing Service\n\n\nYou can get a thesis statement for any essay that you want, and it will help you to organize your ideas.\xa0\n\nWith our thesis statement writing service, you will be able to get a paper that is original and unique.\xa0\n\n\nOur thesis statement writing service will help you to write a thesis statement that will help you to organize your ideas.\xa0\n\nWe will make sure that you get a paper that is original and unique.\xa0\n\nOur Thesis Statement Writing Service\n\nWhat are the benefits of writing a thesis statement?\n\nThere are several benefits of writing a thesis statement. Here are a few of them:\n\n• It helps you to organize your ideas.\n• It'}]
```

## Mask filling

This pipeline, as the name suggests, fills in masks, which are blank spots, in a text.

Now with an example :

```python
classifier = pipeline("fill-mask")
classifier(
    "The capital of France is <mask>.",
    top_k=5
)
```

We get :

```python
[{'score': 0.2703714668750763, 'token': 2201, 'token_str': ' Paris', 'sequence': 'The capital of France is Paris.'}, {'score': 0.0558835007250309, 'token': 12790, 'token_str': ' Lyon', 'sequence': 'The capital of France is Lyon.'}, {'score': 0.029898053035140038, 'token': 4612, 'token_str': ' Barcelona', 'sequence': 'The capital of France is Barcelona.'}, {'score': 0.023081665858626366, 'token': 12696, 'token_str': ' Monaco', 'sequence': 'The capital of France is Monaco.'}, {'score': 0.02097989059984684, 'token': 5459, 'token_str': ' Berlin', 'sequence': 'The capital of France is Berlin.'}]
```

The `top_k` argument found above defines how many possibilities we want.

> ⚠️ **Note:** Here the \<mask> is what the model replaces, also known as *mask token*. Depending on the model, this can change, so it's always good practice to check what the *mask token* for a model is when using it.

## Named entity recognition

The **Named entity recognition** (or **NER** for short) is a task where the model has to find which parts of a given text represent people, locations or organizations.

With the pipeline defined :
```python
ner = pipeline("ner", grouped_entities=True)
output = ner(
    "Hugging Face is based in New York City.",
)
```

We get :

```python
[{'entity_group': 'ORG', 'score': 0.8907566, 'word': 'Hugging Face', 'start': 0, 'end': 12}, {'entity_group': 'LOC', 'score': 0.9991805, 'word': 'New York City', 'start': 25, 'end': 38}]
```

Here I pass the `grouped_entities` as `true`, which tells the model to regroup parts of the same entity together.

As stated before, the model can identifie a person (PER), a location (LOC) and an organization (ORG).

## Question answering

This pipeline answers a question using information that was provided as context.

When running :

```python
classifier = pipeline(question-answering)
output = classifier(
    question="What is the capital of France?",
    context="Paris is the capital of France."
)
```

We get :

```python
{'score': 0.9960771203041077, 'start': 0, 'end': 5, 'answer': 'Paris'}
```

> ℹ️ **Info:** This pipeline only provides the answer from the context given, it does not generate new text.

## Summarization

This task has the model reduce a text into a shorter text while keeping all (or almost all) of the important aspects refered in the text.

We can set the minimum and maximum amount of words the text can have with `min_length` and `max_length` respectively. We can also set if we want the model to be more or less creative with the argument `do_sample`. If set to `False`, the model will use greedy decoding (always picking the most likely next token). If set to `True` however, the model will choose from the probability distribution of possible next tokens, making the answers more diverse and creative.

Running the model like this :

```python
classifier = pipeline("summarization", model="facebook/bart-large-cnn")
output = classifier(
    '''Hugging Face is a company that provides tools and services for natural language processing. 
    They are known for their open-source libraries and models, which have become widely used in the AI community. 
    Hugging Face also offers a platform for hosting and sharing models, making it easier for developers to access and use state-of-the-art AI technologies. 
    Their mission is to democratize AI and make it accessible to everyone.''',
    max_length=50,
    min_length=25,
)
```

We get :

```python
[{'summary_text': 'Hugging Face is a company that provides tools and services for natural language processing. They are known for their open-source libraries and models, which have become widely used in the AI community.'}]
```

## Translation

When translating, there are 2 available options. The first one is stating in the pipeline which translation you want, like `translation_en_to_fr`. However, it is advisable to instead pick a model that was trained to perform that specific translation.

Creating the pipeline and the output :

```python
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
output = translator("Ce cours est produit par Hugging Face.")
```

We get : 

```python
[{'translation_text': 'This course is produced by Hugging Face.'}]
```

## Image and audio pipelines

Another amazing thing about transformers is how they can also perform tasks with images and audio.

### Image Classification

This pipeline classifies an image.

Running this pipeline :

```python
image_classifier = pipeline(
    task="image-classification", model="google/vit-base-patch16-224"
)
result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
```

I got :

```python
[{'label': 'lynx, catamount', 'score': 0.4334995746612549}, {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'score': 0.03479621559381485}, {'label': 'snow leopard, ounce, Panthera uncia', 'score': 0.03240194916725159}, {'label': 'Egyptian cat', 'score': 0.023944787681102753}, {'label': 'tiger cat', 'score': 0.022889269515872}]
```

### Automatic speech recognition

This pipeline recognises text from an audio.

Running : 

```python
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
result = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
```

I got :

```python
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

## Combining data from multiple sources

A great thing about transformers is how they can combine and process a variety of different data types (audio, text, images, etc...). 

This is especially useful when you need to:

1. Search across multiple databases or repositories
2. Consolidate information from different formats (text, images, audio)
3. Create a unified view of related information

For example, you could build a system that:
- Searches for information across databases in multiple modalities like text and image.
- Combines results from different sources into a single coherent response. For example, from an audio file and text description.
- Presents the most relevant information from a database of documents and metadata.

---

➡️ [Next: Transformer Architecture and History](04_transformer_architecture.md)