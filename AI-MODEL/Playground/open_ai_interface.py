import os
import openai
import json
# openai.organization = "Personal"
openai.api_key = "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"

model_list = openai.Model.list()
for model in model_list['data']:
    print(model['id'])


completion = openai.Completion.create(model="gpt-3.5-turbo-0301", prompt="Hello world")
