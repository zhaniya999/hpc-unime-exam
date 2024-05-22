import json

class PathsMessage:
    def __init__(self):
        self.a=[]
        self.b=None
    
    def addPathA(self,path):
        self.a.append(path)
    
    def addPathB(self,path):
        self.b =path

    def getPathA(self):
        return self.a
    
    def getPathB(self):
        return self.b
    
    def haveData(self):
        return len(self.a)>0

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__)
    
    def load_from_json(cls, json_string):
        return json.loads(json_string, object_hook=cls)


class GenericMessage:
    def __init__(self, row):
        self.row = row
    
    def setRow(self,row):
        self.row = row
    
    def getRow(self):
        return self.row

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__)
    
    def load_from_json(cls, json_string):
        return json.loads(json_string, object_hook=cls)

class DataMessage(GenericMessage):
    
    def __init__(self, row, a, b):
        self.row = row
        self.a = a
        self.b = b
    
    def setA(self,a):
        self.a = a
    
    def setB(self,b):
        self.b = b

    def getA(self):
        return self.a
    
    def getB(self):
        return self.b
    
class ResponseMessage(GenericMessage):
    
    def __init__(self, row, result, type, time):
        self.row = row
        self.result = result
        self.time = time
        self.type = type
    
    def setResult(self,result):
        self.result = result
    
    def setType(self,type):
        self.type = type
    
    def setTime(self,time):
        self.time = time
    
    def getResult(self):
        return self.result
    
    def getType(self):
        return self.type
    
    def getTime(self):
        return self.time