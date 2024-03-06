import os
import shutil
import argparse
import logging
import time
import getpass
import sys
import csv
import tensorflow as tf

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_dir(dir_name, clear=True):
    if os.path.exists(dir_name):
        if clear:
            try:
                shutil.rmtree(dir_name)
            except:
                pass
            try:
                os.makedirs(dir_name)
            except:
                pass
    else:
        try:
            os.makedirs(dir_name)
        except:
            pass


def dir_ls(dir_path):
    dir_list = os.listdir(dir_path)
    dir_list.sort()
    return dir_list


def system_pause():
    getpass.getpass("Press Enter to Continue")


def get_arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def remove_color(key):
    for i in range(len(key)):
        if key[i] == '@':
            return key[:i]
    return key


class Logger:
    def __init__(self, name):
        self.name = name
        self.id = time.strftime('%Y%m%d-%H%M%S')
        self.my_log_dir = f'log/{name}/{self.id}/'

        log_file = self.my_log_dir + 'out.log'

        make_dir('log', clear=False)
        make_dir('log/' + name, clear=False)
        make_dir(self.my_log_dir, clear=False)

        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

        self.summary_writer = tf.summary.create_file_writer(self.my_log_dir + 'model')

        self.data = {}

    def debug(self, *args):
        self.logger.debug(*args)

    def info(self, *args):
        self.logger.info(*args)  # default level

    def warning(self, *args):
        self.logger.warning(*args)

    def error(self, *args):
        self.logger.error(*args)

    def critical(self, *args):
        self.logger.critical(*args)

    def add_item(self, key):
        self.data[key] = []

    def add_record(self, key, value):
        self.data[key].append(value)

    def tabular_show(self, iter):
        row_format = "{:<20}" * (len(self.data))
        if iter % 10 == 0:
            self.info('\n')
            self.info(row_format.format(list(self.data.keys())[0], *list(self.data.keys())[1:]))
        self.info(row_format.format(list(self.data.values())[0][-1], *[val[-1] for val in list(self.data.values())[1:]]))

    def save_csv(self, filename='progress'):
        with open(self.my_log_dir + filename + '.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.data.keys())
            writer.writerows(zip(*self.data.values()))

def get_logger(name):
    return Logger(name)
