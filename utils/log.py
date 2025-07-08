#!/usr/bin/env python3

from datetime import date

class log_writer():
    def __init__(self):
        self.time_str = date.today().strftime("%Y-%m-%d")

    def print_write_log(self, *args):
        msg = ' '.join(str(arg) for arg in args)  # 拼接多个参数为字符串
        print(msg)
        with open(self.time_str + ".log", "a", encoding="utf-8") as f:
            f.write(msg)

if __name__ == '__main__':
    log_writer = log_writer()
    log_writer.print_write_log("some doc\n")

    try:
        1 / 0
    except Exception as error:
        exce_counter = 0
        if exce_counter == 0:
            log_writer.print_write_log(error, '\n')

    log_writer.print_write_log('%d exceptions encountered during last training' % 100, "\nwaht")
