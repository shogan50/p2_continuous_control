import matplotlib as plt
import os
import sys

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
            line = line + str(arg)
        f.write(line +'\n')
        print(line)
        f.close()

    def add_line(self, *argv):
        line = ''
        for arg in argv:
            line = line + str(arg)
        self.lines.append(line + '\n')

    def save_lines(self):
        f = open(self.log_path, mode = 'a')
        for line in self.lines:
            f.write(line)
            print(line)
        f.close()
        self.lines = []

    def add_kwargs(self,kwargs):
        line = []
        for arg in kwargs.items():
            self.lines.append(str(arg))
        self.save_lines()

class Logger(object):
    def __init__(self, f_name):
        self.terminal = sys.stdout
        self.log = open(f_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


