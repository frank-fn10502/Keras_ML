



class IDataset:
    def __init__(self) -> None:
        self.trainData = None
        self.validationData = None
        self.datasetName = type(self).__name__
        self.inputShape = (32,32,1)
        self.classes = 10