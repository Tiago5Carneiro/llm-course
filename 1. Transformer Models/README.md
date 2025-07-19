# Tranformer Models

## Introduction 

The first thing to keep in mind is the difference between NLP and LLMs, which are both taught in this course.

Starting with **NLP** or **Natural Language Processing** :

- Is the broader field 
- Focused on enabling computers to understand, interpret and generate human language
- Encompasses techniques like sentiment analysis, named entity recognition and machine translation

Now touching on **LLMs** or **Large Language Models** :

- Subset of NLP models
- Massive size and extensive training data 
- Ability to perform a wide range of language tasks with minimal task-specific training
- Some examples : Llama, GPT, Claude, Gemini, Mistral, etc...

What I will be learning in this course :

![What to expect](../media/what_to_expect.png)

##  Natural Language Processing and Large Language Models

NLP is a field of linguistics and machine learning, focused on understanding human language, with all it's nuance.

Common **NLP** tasks with examples :

- Classifying whole sentences : Detecting if an email is spam, determining if a sentence is grammatically correct.
- Classifying each word in a sentence : Identifying nouns, verbs, adjectives, or named entities like people, locations, organizations, etc..
- Generating text context : Filling in texts with masked words, compliting a prompt with auto-generated text.
- Extracting and answer from a text : Provided a question and a context, answering the given question.
- Generating a new sentence from an input text : Translating a text from another language or creating a summary of a given text.

###  The Rise of Large Language Models (LLMs)

Recently **LLMs** (which are a subset of NLPs) have revolutionized the field. They can be **characterized** by:

- Scale : Huge ammounts of parameters (millions, billions or even hundreds of billions).
- General capabilities : Can perform multiple tasks without task-specific training.
- In-context learning : They perform tasks from the context given in a prompt.
- Emergent abilities : As the models grow, new capabilities appear that weren't explicitly programmed or anticipated.

Even with all these advantages, LLMs have their **limitations** : 

- Hallucinations : They can generate incorrect information confidently 
- Lack of true understanding : In essence, these models are made to predict, only operating on purely statistical patterns.
- Bias : They may represent bias found in trainning data or inputs.
- Context windows : Their context is limited by the size of the context windows.
- Computational resources : A lot of resources are required for the computations that LLMs do.

## Transformers, what can they do?

Firstly, what even is a transformer?

**Transformer** is a neural network arquitecture invented by Google in 2017. It's a core component in many modern LLMs.

HuggingFace's opensource library contains a Transformers library, which I will be using.

Let's start with the 
```python 
pipeline()
```
function, this function returns an object. It connects a model with it's necessary preprocessing and postprocessing steps, which allows us to input text and get an intelligible answer. It is the most high-level API, as all the necessary steps to transform the input into readable data for the model, the postprocessing necessary for human comprehension and the actual model computation are performed with it.

There are plenty of available pipelines :

#### Text pipelines

- `text-generation`: Generate text from a prompt
- `text-classification`: Classify text into predefined categories
- `summarization`: Create a shorter version of a text while preserving key information
- `translation`: Translate text from one language to another
- `zero-shot-classification`: Classify text without prior training on specific labels
- `feature-extraction`: Extract vector representations of text

#### Image pipelines

- `image-to-text`: Generate text descriptions of images
- `image-classification`: Identify objects in an image
- `object-detection`: Locate and identify objects in images

#### Audio pipelines

- `automatic-speech-recognition`: Convert speech to text
- `audio-classification`: Classify audio into categories
- `text-to-speech`: Convert text to spoken audio

#### Multimodal pipelines

- `image-text-to-text`: Respond to an image based on a text prompt

We will analyse how some of these pipelines work, what their

### Sentiment Analysis 

This is one of the available pipeline available in the transformers. It returns wether a sentence is 'POSITIVE' or 'NEGATIVE', with the associated confidence percentage.

#### First exercise

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

### Zero-Shot Classification

Another one of the available pipelines is the zero-shot classification. In this pipeline, we provide a text, as well as a list of candidate labs that we wan't the model to associate with.

With the following input :

```python
(
    "Portugal joined nato in 1949.",
    candidate_labels=["education", "politics", "buisiness", "sports"]
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

### Text generation

For text generation, the main idea is that, provided with an initial prompt, the model auto-completes it by generating the remaining text.