from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizers = AutoTokenizer.from_pretrained("bigscience/bloom")

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom",
    device_map="auto",
    torch_dtype="auto"
)

prompt = "Generate a dataset to train a financial question answering AI. The dataset should be in json format, and each" \
         "item in the dataset should appear as follows" \
         "Question: (Actual question user is asking the AI, IE 'What were th important points in apples earning statements last quarter" \
         "Program: should be a short python program which gathers data needed to process the question" \
         "Answer: Expected answer to the question"
