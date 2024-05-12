class DataMessage:
    
    def __init__(self, row, a, b):
        self.row = row
        self.a = a
        self.b = b
    
    def setRow(self,row):
        self.row = row
    
    def setA(self,a):
        self.a = a
    
    def setB(self,b):
        self.b = b

    def getRow(self):
        return self.row
    
    def getA(self):
        return self.a
    
    def getB(self):
        return self.b
    
class ResponseMessage:
    
    def __init__(self, row, result, type, time):
        self.row = row
        self.result = result
        self.time = time
        self.type = type
    
    def setRow(self,row):
        self.row = row
    
    def setResult(self,result):
        self.result = result
    
    def setType(self,type):
        self.type = type
    
    def setTime(self,time):
        self.time = time
    
    def getRow(self):
        return self.row
    
    def getResult(self):
        return self.result
    
    def getType(self):
        return self.type
    
    def getTime(self):
        return self.time