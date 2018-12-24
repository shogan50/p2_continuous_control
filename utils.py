import matplotlib as plt
import os

class log_me:
    def __init__(self):
        self.lines = []
        self.cwd = os.getcwd()
        self.f_name = 'run log '
        id = 0

        while os.path.exists(self.cwd + os.sep + self.f_name + str(id) + '.log'):
            id += 1
        self.log_path = self.cwd + os.sep + self.f_name + str(id) + '.log'

    def log(self, str):
        f = open(self.log_path, mode='a')
        f.write(str + '\n')
        print(str)
        f.close()

    def add_line(self, str):
        self.lines.append(str)

    def save_lines(self):
        f = open(self.log_path, mode = 'a')
        for line in self.lines:
            f.write(line + '\n')
            print(line)
        f.close()



