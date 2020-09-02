
import time
from threading import *


class Hello(Thread):
    def run(self):
        for i in range(5):
            print("Hello")
            time.sleep(1)



class Hi(Thread):
    def run(self):
        for i in range(5):
            print("Hi")


t1 = Hello()
t2 = Hi()

t1.start()   #internally start() will call run()
t2.start()

t1.join()
t2.join()

print("Bye")