@echo off
echo Checking free space...
dir C:\
python %~dp0super_dataset_builder.py --count 100000
