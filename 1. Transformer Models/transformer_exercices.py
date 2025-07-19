from transformers import pipeline

classfier = None

def set_classifier(type):
    if classfier is None:
        global classifier
        classifier = pipeline(type)

def exercise_1():
    output = classifier("I'm so excited for the upcoming weekend")

    print(output)

def exercise_2():
    output = classifier(
        ["I am not happy with the service I received.", "That meal was delicious!"]
        )
    print(output)

def main():
    set_classifier("sentiment-analysis")
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

main()