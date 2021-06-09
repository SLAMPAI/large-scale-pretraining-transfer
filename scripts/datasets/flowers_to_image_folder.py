import os
import shutil
for split in ('mod_train', 'mod_test'):
    for f in open(split+'.txt').readlines():
        f = f.strip()
        fname, class_name = f.split(' ')
        class_folder = os.path.join(split, class_name)
        os.makedirs(class_folder, exist_ok=True)
        F = os.path.join(class_folder, os.path.basename(fname))
        shutil.copyfile(fname,F)
