
import psutil
import os

def print_memory_diags():
    ''' Print memory diagnostics including the active resident set size'''
    process = psutil.Process(os.getpid())
    print("Memory usage: {:.3f} GB".format(process.memory_info().rss/1000000000.0))
