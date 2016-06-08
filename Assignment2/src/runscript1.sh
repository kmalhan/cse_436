#! /bin/bash

# This script to run all test cases for assignment 2
# Usage: runscript1.sh [filename]

# Assume sum.c is sum, and mm.c is mm

if [ -e $1 ]; then
  echo "File $1 already exists!"
else
  echo > $1
fi

echo "Assignment2 Test Run" >> $1
echo >> $1

echo "==Platform that code is running==" >> $1
uname -a >> $1

# Set environment variable for scheduling method.
export OMP_SCHEDULE="static,0"

# sum.c different thread number runs
echo >> $1
echo "================ sum.c thread = 1 =================="
echo "================ sum.c thread = 1 ==================" >> $1
echo >> $1

bin/sum 10000000 1 >> $1

echo >> $1
echo "================ sum.c thread = 2 =================="
echo "================ sum.c thread = 2 ==================" >> $1
echo >> $1

bin/sum 10000000 2 >> $1

echo >> $1
echo "================ sum.c thread = 4 =================="
echo "================ sum.c thread = 4 ==================" >> $1
echo >> $1

bin/sum 10000000 4 >> $1

echo >> $1
echo "================ sum.c thread = 8 =================="
echo "================ sum.c thread = 8 ==================" >> $1
echo >> $1

bin/sum 10000000 8 >> $1

# mm.c different number of threads runs
echo >> $1
echo "================ mm.c thread = 1 =================="
echo "================ mm.c thread = 1 ==================" >> $1
echo >> $1

bin/mm 1024 1024 1024 1 >> $1

echo >> $1
echo "================ mm.c thread = 2 =================="
echo "================ mm.c thread = 2 ==================" >> $1
echo >> $1

bin/mm 1024 1024 1024 2 >> $1

echo >> $1
echo "================ mm.c thread = 4 =================="
echo "================ mm.c thread = 4 ==================" >> $1
echo >> $1

bin/mm 1024 1024 1024 4 >> $1

echo >> $1
echo "================ mm.c thread = 8 =================="
echo "================ mm.c thread = 8 ==================" >> $1
echo >> $1

bin/mm 1024 1024 1024 8 >> $1

echo >> $1
echo "=============== done =============="
