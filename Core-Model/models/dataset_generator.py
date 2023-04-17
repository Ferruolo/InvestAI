from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz-1b1"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

prompt = "Generate a dataset to train a financial question answering AI. These questions should concern relavent metrics, data availible in 10ks, along with simply defining terms and filling basic knowledge" \
         "The dataset should be in json format, and eachitem in the dataset should appear as follows\n" \
         "Question: (Actual question user is asking the AI, IE 'What were the important points in apples earning statements last quarter\n" \
         "Program: should be a short python program which gathers data needed to process the question\n" \
         "Answer: Expected answer to the question\n" \
         "Generate 1 element of this dataset"


print(prompt)




inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_length=10000)
print([tokenizer.decode(x.to('cpu')) for x in outputs])
