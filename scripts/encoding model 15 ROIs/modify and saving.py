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

for ii,subject in enumerate(all_subjects):
    subject
    with open(template.replace('.py',' ips {}.py'.format(ii + 1)),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "sub                     = '*'" in line:
                    line = line.replace("sub                     = '*'","sub                     = '{}'".format(subject))
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

# This is a script to send {template.replace('.py',' ips {}.py'.format(ii + 1))} as a batch job.
# it works on subject {subject}

#$ -cwd
#$ -o test_run/out_{ii + 1}.txt
#$ -e test_run/err_{ii + 1}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "{qsub_key}_{ii+1}"
#$ -S /bin/bash

module load rocks-python-3.6
python "{template.replace('.py',' ips {}.py'.format(ii + 1))}"
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
            f.write('time.sleep(30)\nos.system("qsub enc_{}")\n'.format(ii+1))

content = '''
#!/bin/bash

# This is a script to send qsub_jobs_en.py as a batch job.

#$ -cwd
#$ -o test_run/out_q.txt
#$ -e test_run/err_q.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N "qsubjobs"
#$ -S /bin/bash

module load rocks-python-3.6
python "qsub_jobs_en.py"
'''
with open('qsub_jobs_en','w') as f:
    f.write(content)


































