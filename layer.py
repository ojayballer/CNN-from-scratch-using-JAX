class Layer:
    def __init__(self):
        self.input=None
        self.output=None

    # I will write .forward() and .backward() here...,to avoid typing the two lines above this line in every class,
    #and raise an error incase .forward() or .backward() are not overrided in the child classes
    def forward(self,input):
        

        raise NotImplementedError("hey!!...you forgot to override the function,thankfully I put this here to make debugging easy!")

    def backward(self,output,learning_rate):


        raise NotImplementedError("please override this function")