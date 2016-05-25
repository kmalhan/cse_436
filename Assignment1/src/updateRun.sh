#! /bin/bash

# This script to run all six cases at once and save result into input argument file.

if [ -e $1 ]; then
  echo "File $1 already exists!"
else
  echo > $1
fi

echo "Assignment1 Test Run" >> $1

echo >> $1
echo "==Platform that code is running==" >> $1
uname -a >> $1
echo >> $1

echo "== Opimization 0 ==" >> $1
echo >> $1
echo "================ Run 1 =================="
echo "================ Run 1 ==================" >> $1
echo >> $1
bin/assignment1_u0 512 512 >> $1
echo >> $1
echo "================ Run 2 =================="
echo "================ Run 2 ==================" >> $1
echo >> $1
bin/assignment1_u0 1024 1024 >> $1
echo >> $1
echo "================ Run 3 =================="
echo "================ Run 3 ==================" >> $1
echo >> $1
bin/assignment1_u0 2048 2048 >> $1
echo >> $1
echo "== Optimication 3 ==" >> $1
echo >> $1
echo "================ Run 4 =================="
echo "================ Run 4 ==================" >> $1
echo >> $1
bin/assignment1_u3 512 512 >> $1
echo >> $1
echo "================ Run 5 =================="
echo "================ Run 5 ==================" >> $1
echo >> $1
bin/assignment1_u3 1024 1024 >> $1
echo >> $1
echo "================ Run 6 =================="
echo "================ Run 6 ==================" >> $1
echo >> $1
bin/assignment1_u3 2048 2048 >> $1
echo >> $1
echo "=============== done =============="
