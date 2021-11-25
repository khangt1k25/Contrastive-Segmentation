#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os


class Path(object):
    """
    User-specific path configuration.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '/content/drive/MyDrive/UCS_local/PASCAL_VOC' # VOC will be automatically downloaded
        db_root_other = '/content/drive/MyDrive/UCS_local/'
        db_names = ['VOCSegmentation', 'MSRCV2']

        if database == '':
            return db_root

        if database == db_names[0]:
            return os.path.join(db_root, database)
        elif database == db_names[1]:
            return os.path.join(db_root_other, database)
        else:
            raise ValueError('Invalid database {}'.format(database))
