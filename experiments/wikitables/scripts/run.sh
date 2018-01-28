#!/bin/bash -e

CLASSPATH="target/scala-2.11/pnp-assembly-0.1.2.jar:lib/stanford-corenlp-3.6.0-models.jar"
echo $CLASSPATH
# Changed from 12 to 30 Gigs
java -Djava.library.path=lib -classpath $CLASSPATH -Xmx30000M "$@" 
