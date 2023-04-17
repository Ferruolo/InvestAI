from models.ticker_classifier import TickerClassifier

class MainProgram:
    def __init__(self):


    def forward(self, q):
        ticker = self.Ticker(q)
        dates = self.dates(q)
        data = self.get_data(dates)



if __name__ == "__main__":
    driver = MainProgram()
    print("Hello, I am a Financial Question Answering Bot. How could I help you today")
    break_condition = True
    while break_condition:
        question = input("Please input a question here, or type EXIT to exit")
        if question == "EXIT":
            break_condition = False
        else:
            response = driver.forward(question)
            print(response)
