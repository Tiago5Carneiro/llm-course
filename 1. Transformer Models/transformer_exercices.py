from transformers import pipeline

classifier = pipeline("sentiment-analysis")
output = classifier("I'm so excited for the upcoming weekend")

print(output)
