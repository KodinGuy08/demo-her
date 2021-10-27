import sys
import threading

class Thread (threading.Thread):
   v = -1

   args = []
   
   def __init__(self, threadID, name, func):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.func = func
      self.status = False

   def print(self, out):
      if not self.v == -1:
         print(out)

   def setParameters(self, args):
      self.args = args
      
   def run(self):
      self.print ("Starting " + self.name)
      self.status = True
      
      self.func(*self.args)
      
      self.print ("Exiting " + self.name)
      self.status = False
