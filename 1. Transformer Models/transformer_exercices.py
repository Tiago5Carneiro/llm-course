from transformers import pipeline

classifier = None

def set_classifier(type):
    global classifier
    if classifier is None:
        classifier = pipeline(type)

def exercise_1():
    output = classifier("I'm so excited for the upcoming weekend")

    print(output)

def exercise_2():
    output = classifier(
        ["I am not happy with the service I received.", "That meal was delicious!"]
        )
    print(output)

def sentiment_main():
    set_classifier("sentiment-analysis")
    running = True
    while running:
        print("Choose one of the following exercises:")
        print("1. Exercise 1")
        print("2. Exercise 2")
        choice = input("Enter the number of your choice: ")
        if choice == "1":
            exercise_1()
        elif choice == "2":
            exercise_2()
        else:
            print("Invalid choice.")

def exercise_3():
    global classifier
    output = classifier(
        "Portugal joined nato in 1949.",
        candidate_labels=["education", "politics", "buisiness", "sports"]
    )
    print(output)

def zero_shot_classification():
    set_classifier("zero-shot-classification")
    running = True
    while running:
        print("Choose one of the following exercises:")
        print("1. Exercise 3")
        print("Q. Quit")
        choice = input("Enter the number of your choice: ")
        if choice == "1":
            exercise_3()
        elif choice.upper() == "Q":
            running = False
        else:
            print("Invalid choice.")

def text_generation():
    global classifier
    set_classifier("text-generation")
    output = classifier(
        "The future of AI is",
        max_length=20,
        num_return_sequences=1
    )
    print(output)

def text_generation_with_model():
    global classifier
    classifier = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
    output = classifier(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )
    print(output)

def mask_filling():
    global classifier
    set_classifier("fill-mask")
    output = classifier(
        "The capital of France is <mask>.",
        top_k=5
    )
    print(output)

def name_entity_recognition():
    ner = pipeline("ner", grouped_entities=True)
    output = ner(
        "Hugging Face is based in New York City.",
    )
    print(output)

def main():
    print("Welcome to the transformer exercises!")
    running = True
    while running:
        print("Choose one of the available pipelines:")
        print("1. Sentiment Analysis")
        print("2. Zero-shot Classification")
        print("3. Text Generation")
        print("4. Text Generation with Model")
        print("5. Mask Filling")
        print("6. Name Entity Recognition")
        print("Q. Quit")
        choice = input("Enter the number of your choice: ")
        if choice == "1":
            sentiment_main()
        elif choice == "2":
            zero_shot_classification()
        elif choice == "3":
            text_generation()
        elif choice == "4":
            text_generation_with_model()
        elif choice == "5":
            mask_filling()
        elif choice == "6":
            name_entity_recognition()
        elif choice.upper() == "Q":
            print("Exiting the program.")
            running = False

main()