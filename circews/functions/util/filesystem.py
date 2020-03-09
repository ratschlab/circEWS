
import os
import os.path

def delete_if_exist(path):
    ''' Deletes a path if it exists on the file-system'''
    if os.path.exists(path):
        os.remove(path)

def create_dir_if_not_exist(path, recursive=False):
    ''' Creates a directory if it does not yet exist in the file system'''
    if not os.path.exists(path):
        if recursive:
            os.makedirs(path)
        else:
            os.mkdir(path)

