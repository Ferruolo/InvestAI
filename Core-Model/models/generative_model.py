from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class RobertaGenerative:
    def __init__(self):
        self.model_name = "deepset/roberta-base-squad2"
        self.nlp = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name)

    def forward(self, question, context):
        return self.nlp({'question': question, 'context': context})
