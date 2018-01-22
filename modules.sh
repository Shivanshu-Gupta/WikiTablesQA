#!/bin/bash

module load apps/binutils/2.25/gnu
module load compiler/gcc/4.9.3/compilervars
module load lib/boost/1.64.0/gnu_ucs4
module load apps/cmake/3.4.1/gnu
module load lib/pcre/8.40/gnu
module load apps/git/2.9.0/gnu

export CC=$(which gcc)
export CXX=$(which g++)
