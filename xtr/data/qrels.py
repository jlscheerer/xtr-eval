class Qrels:
    def __init__(self, data):
        super().__init__()
        self.data = data

    @staticmethod
    def cast(data):
        return Qrels(data=data)