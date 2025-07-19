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

### First exercise

For the [first exercise](transformer_exercices.py#10), we run a sentiment analysis on the sentence

```python
"I'm so excited for the upcoming weekend"
```

Which gave the following result :

```python
[{'label': 'POSITIVE', 'score': 0.9996603727340698}]
```

Multiple sentence is also available as you can see with [exercise 2](transformer_exercices.py#15). These sentences :

```python
["I am not happy with the service I received.", "That meal was delicious!"]
```

Generated the following output :

```python
[{'label': 'NEGATIVE', 'score': 0.9994329810142517}, {'label': 'POSITIVE', 'score': 0.9998834133148193}]
```