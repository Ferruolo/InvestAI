from transformers import pipeline
import re

# Define the sentence to extract information from
sentence = "What important points were given in Apple and Amazon's earnings reports last week"

# Use the Hugging Face named entity recognition (NER) pipeline to extract the entities
ner_pipeline = pipeline("ner", grouped_entities=True)
ner_result = ner_pipeline(sentence)

# Get the entities from the result
entities = [entity["word"] for entity in ner_result if entity["entity_group"] == "ORG"]

# Use regular expressions to extract the data points and date
data_points_pattern = re.compile(r"(important points|key points|highlights|main takeaways)")
date_pattern = re.compile(r"last week")

data_points_result = data_points_pattern.search(sentence)
date_result = date_pattern.search(sentence)

# Get the data points and date from the result
data_points = data_points_result.group()
date = date_result.group()

# Print out the extracted information
print("Tasks: Summarize the", data_points, "in the earnings reports for", ", and ".join(entities) + ".")
print("Entities:", ", ".join(entities) + ".")
print("Data Points:", data_points + ".")
print("Dates:", date + ".")
