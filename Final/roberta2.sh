#!/bin/bash
### Note: No commands may be executed until after the #PBS lines
### Account information
### PBS -W group_list=zelili_1 -A zelili_1
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N testlzl6
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e testlzl6.err
#PBS -o testlzl6.log
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=8:gpus=1
### Memory
#PBS -l mem=60gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 6 hours)
#PBS -l walltime=8:00:00
/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python3.6 /home/people/zelili/ds_p/final/roberta2.py