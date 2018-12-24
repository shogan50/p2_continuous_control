import matplotlib as plt
import os

class log_me:
    def __init__(self,f_name):
        self.lines = []
        self.cwd = os.getcwd()
        self.f_name = f_name
        id = 0
        self.log_path = self.cwd + os.sep + self.f_name

    def log(self, *argv):
        f = open(self.log_path, mode='a')
        line = ''
        for arg in argv:
            line = line + arg
        f.write(line)
        print(line)
        f.close()

    def add_line(self, *argv):
        line = ''
        for arg in argv:
            line = line + arg
        self.lines.append(line)

    def save_lines(self):
        f = open(self.log_path, mode = 'a')
        for line in self.lines:
            f.write(line)
            print(line)
        f.close()



