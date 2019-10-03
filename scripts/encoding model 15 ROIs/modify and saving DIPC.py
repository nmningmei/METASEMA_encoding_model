#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:42:53 2019

@author: nmei
"""

import os
from glob import glob
import numpy as np

experiment              = 'metasema'
qsub_key = 'msen15'
working_dir             = '../../../../../{}/preprocessed_uncombined_with_invariant/'.format(experiment) # where the data locates
all_files = glob(os.path.join(working_dir,'*','*.csv'))
all_subjects = np.unique([item.split('/')[-2] for item in all_files])

template = 'encoding model 15 ROIs.py'
n_jobs = 12
core = 12
mem = core * 5
cput = 12 * core
for ii,subject in enumerate(all_subjects):
    subject
    with open(template.replace('.py',' DIPC {}.py'.format(ii + 1)),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "sub                     = '*'" in line:
                    line = line.replace("sub                     = '*'","sub                     = '{}'".format(subject))
                    new_file.write(line)
                elif "n_jobs                  = 1 # " in line:
                    line = line.replace('1',f'{n_jobs}')
                    new_file.write(line)
                else:
                    new_file.write(line)
            old_file.close()
        new_file.close()

if not os.path.exists('test_run'):
    os.mkdir('test_run')
else:
    [os.remove('test_run/'+f) for f in os.listdir('test_run')]

for ii,subject in enumerate(all_subjects):
    subject
    content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes=1:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N msen15f{ii+1}
#PBS -M nmei@bcbl.eu
#PBS -o test_run/out_{subject}.txt
#PBS -e test_run/err_{subject}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:$PATH"
pwd
python "{template.replace('.py',' DIPC {}.py'.format(ii + 1))}"
"""
    with open('enc_{}'.format(ii + 1),'w') as f:
        f.write(content)

content = '''
import os
import time
'''
with open('qsub_jobs_en.py','w') as f:
    f.write(content)

with open('qsub_jobs_en.py','a') as f:
    for ii,sub_name in enumerate(all_subjects):
        if ii == 0:
            f.write('os.system("qsub enc_{}")\n'.format(ii+1))
        else:
            f.write('time.sleep(3)\nos.system("qsub enc_{}")\n'.format(ii+1))




































