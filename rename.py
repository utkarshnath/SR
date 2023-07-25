import os
import shutil
import sys
def rename_files_in_subdir(parent_dir):
    for dir_name in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    new_name = dir_name + '_' + filename
                    new_path = os.path.join(subdir_path, new_name)
                    shutil.move(file_path, new_path)

# usage
rename_files_in_subdir(sys.argv[1])
