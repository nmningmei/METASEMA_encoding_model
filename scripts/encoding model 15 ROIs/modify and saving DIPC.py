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
working_dir             = '../../../../data/{}/preprocessed_uncombined_with_invariant/'.format(experiment) # where the data locates
all_files = glob(os.path.join(working_dir,'*','*.csv'))
all_subjects = np.unique([item.split('/')[-2] for item in all_files])

template = 'encoding model 15 ROIs.py'
n_jobs = -1
core = 16
mem = core * 5
cput = 24 * core
for ii,subject in enumerate(all_subjects):
    subject
    with open(template.replace('.py',' DIPC {}.py'.format(ii + 1)),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "star means all subjects" in line:
                    line = line.replace('*','{}'.format(subject))
                    
                new_file.write(line)
            old_file.close()
        new_file.close()

if not os.path.exists('outputs'):
    os.mkdir('outputs')
else:
    [os.remove('outputs/'+f) for f in os.listdir('outputs')]

for ii,subject in enumerate(all_subjects):
    subject
    content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={1}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N {qsub_key}_{ii+1}
#PBS -o outputs/out_{ii+1}.txt
#PBS -e outputs/err_{ii+1}.txt

cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
python "{template.replace('.py',' DIPC {}.py'.format(ii + 1))}"
"""
    with open('enc_{}'.format(ii + 1),'w') as f:
        print(content)
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
            f.write('time.sleep(1)\nos.system("qsub enc_{}")\n'.format(ii+1))

with open('run_all.py','w') as f:
    f.write("""import os\n""")
    f.close()

with open('run_all.py','a') as f:
    for item in glob('encoding*DIPC*py'):
        f.write(f'''os.system('python "{item}"')\n''')
    f.close()


































