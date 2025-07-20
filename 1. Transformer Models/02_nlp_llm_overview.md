#  Natural Language Processing and Large Language Models

NLP is a field of linguistics and machine learning, focused on understanding human language, with all it's nuance.

Common **NLP** tasks with examples :

- Classifying whole sentences : Detecting if an email is spam, determining if a sentence is grammatically correct.
- Classifying each word in a sentence : Identifying nouns, verbs, adjectives, or named entities like people, locations, organizations, etc..
- Generating text context : Filling in texts with masked words, compliting a prompt with auto-generated text.
- Extracting and answer from a text : Provided a question and a context, answering the given question.
- Generating a new sentence from an input text : Translating a text from another language or creating a summary of a given text.

##  The Rise of Large Language Models (LLMs)

Recently **LLMs** (which are a subset of NLPs) have revolutionized the field. They can be **characterized** by:

- Scale : Huge amounts of parameters (millions, billions or even hundreds of billions).
- General capabilities : Can perform multiple tasks without task-specific training.
- In-context learning : They perform tasks from the context given in a prompt.
- Emergent abilities : As the models grow, new capabilities appear that weren't explicitly programmed or anticipated.

Even with all these advantages, LLMs have their **limitations** : 

- Hallucinations : They can generate incorrect information confidently 
- Lack of true understanding : In essence, these models are made to predict, only operating on purely statistical patterns.
- Bias : They may represent bias found in trainning data or inputs.
- Context windows : Their context is limited by the size of the context windows.
- Computational resources : A lot of resources are required for the computations that LLMs do.

---

➡️ [Next: Transformer Pipelines](03_transformer_pipelines.md)