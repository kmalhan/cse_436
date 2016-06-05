#! /bin/bash

# This script to run all test cases for assignment 2
# Usage: runscript1.sh [filename]

# For different schedule policy

# sum.c different scheduling method test

if [ -e $1 ]; then
  echo "File $1 already exists!"
else
  echo > $1
fi

echo "Assignment2 Test Run" >> $1
echo >> $1

echo "==Platform that code is running==" >> $1
uname -a >> $1

#####################################################################################
export OMP_SCHEDULE="static,0"
echo >> $1
echo "================ sum.c static =================="
echo "================ sum.c static, thread = 1 ==================" >> $1
echo >> $1

bin/sum 10000000 1 >> $1

echo >> $1
echo "================ sum.c static, thread = 2 ==================" >> $1
echo >> $1

bin/sum 10000000 2 >> $1

echo >> $1
echo "================ sum.c static, thread = 4 ==================" >> $1
echo >> $1

bin/sum 10000000 4 >> $1

echo >> $1
echo "================ sum.c static, thread = 8 ==================" >> $1
echo >> $1

bin/sum 10000000 8 >> $1

#################################################################################
export OMP_SCHEDULE="static,2000"
echo >> $1
echo "================ sum.c static, 2000 =================="
echo "================ sum.c static, 2000, thread = 1 ==================" >> $1
echo >> $1

bin/sum 10000000 1 >> $1

echo >> $1
echo "================ sum.c static, 2000, thread = 2 ==================" >> $1
echo >> $1

bin/sum 10000000 2 >> $1

echo >> $1
echo "================ sum.c static, 2000, thread = 4 ==================" >> $1
echo >> $1

bin/sum 10000000 4 >> $1

echo >> $1
echo "================ sum.c static, 2000, thread = 8 ==================" >> $1
echo >> $1

bin/sum 10000000 8 >> $1

###################################################################################
export OMP_SCHEDULE="static,200"
echo >> $1
echo "================ sum.c static, 200 =================="
echo "================ sum.c static, 200, thread = 1 ==================" >> $1
echo >> $1

bin/sum 10000000 1 >> $1

echo >> $1
echo "================ sum.c static, 200, thread = 2 ==================" >> $1
echo >> $1

bin/sum 10000000 2 >> $1

echo >> $1
echo "================ sum.c static, 200, thread = 4 ==================" >> $1
echo >> $1

bin/sum 10000000 4 >> $1

echo >> $1
echo "================ sum.c static, 200, thread = 8 ==================" >> $1
echo >> $1

bin/sum 10000000 8 >> $1

###################################################################################
export OMP_SCHEDULE="static,20"
echo >> $1
echo "================ sum.c static, 20 =================="
echo "================ sum.c static, 20, thread = 1 ==================" >> $1
echo >> $1

bin/sum 10000000 1 >> $1

echo >> $1
echo "================ sum.c static, 20, thread = 2 ==================" >> $1
echo >> $1

bin/sum 10000000 2 >> $1

echo >> $1
echo "================ sum.c static, 20, thread = 4 ==================" >> $1
echo >> $1

bin/sum 10000000 4 >> $1

echo >> $1
echo "================ sum.c static, 20, thread = 8 ==================" >> $1
echo >> $1

bin/sum 10000000 8 >> $1

##################################################################################
export OMP_SCHEDULE="dynamic,2000"
echo >> $1
echo "================ sum.c dynamic, 2000 =================="
echo "================ sum.c dynamic, 2000, thread = 1 ==================" >> $1
echo >> $1

bin/sum 10000000 1 >> $1

echo >> $1
echo "================ sum.c dynamic, 2000, thread = 2 ==================" >> $1
echo >> $1

bin/sum 10000000 2 >> $1

echo >> $1
echo "================ sum.c dynamic, 2000, thread = 4 ==================" >> $1
echo >> $1

bin/sum 10000000 4 >> $1

echo >> $1
echo "================ sum.c dynamic, 2000, thread = 8 ==================" >> $1
echo >> $1

bin/sum 10000000 8 >> $1

####################################################################################
export OMP_SCHEDULE="dynamic,200"
echo >> $1
echo "================ sum.c dynamic, 200 =================="
echo "================ sum.c dynamic, 200, thread = 1 ==================" >> $1
echo >> $1

bin/sum 10000000 1 >> $1

echo >> $1
echo "================ sum.c dynamic, 200, thread = 2 ==================" >> $1
echo >> $1

bin/sum 10000000 2 >> $1

echo >> $1
echo "================ sum.c dynamic, 200, thread = 4 ==================" >> $1
echo >> $1

bin/sum 10000000 4 >> $1

echo >> $1
echo "================ sum.c dynamic, 200, thread = 8 ==================" >> $1
echo >> $1

bin/sum 10000000 8 >> $1

#####################################################################################
export OMP_SCHEDULE="dynamic,20"
echo >> $1
echo "================ sum.c dynamic, 20 =================="
echo "================ sum.c dynamic, 20, thread = 1 ==================" >> $1
echo >> $1

bin/sum 10000000 1 >> $1

echo >> $1
echo "================ sum.c dynamic, 20, thread = 2 ==================" >> $1
echo >> $1

bin/sum 10000000 2 >> $1

echo >> $1
echo "================ sum.c dynamic, 20, thread = 4 ==================" >> $1
echo >> $1

bin/sum 10000000 4 >> $1

echo >> $1
echo "================ sum.c dynamic, 20, thread = 8 ==================" >> $1
echo >> $1

bin/sum 10000000 8 >> $1

####################################################################################
export OMP_SCHEDULE="guided,200000"
echo >> $1
echo "================ sum.c guided, 200000 =================="
echo "================ sum.c guided, 200000, thread = 1 ==================" >> $1
echo >> $1

bin/sum 10000000 1 >> $1

echo >> $1
echo "================ sum.c guided, 200000, thread = 2 ==================" >> $1
echo >> $1

bin/sum 10000000 2 >> $1

echo >> $1
echo "================ sum.c guided, 200000, thread = 4 ==================" >> $1
echo >> $1

bin/sum 10000000 4 >> $1

echo >> $1
echo "================ sum.c guided, 200000, thread = 8 ==================" >> $1
echo >> $1

bin/sum 10000000 8 >> $1

echo >> $1
echo "=============== Done ========================"
