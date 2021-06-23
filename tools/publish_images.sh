#!/bin/bash

NAME=${1:-test}
SRC=${2:-/mnt/nfs/scratch1/wenlongzhao/roosts_data}
HOST=${3:-doppler.cs.umass.edu}
DST=${4:-/var/www/html/roost/img}

ssh $HOST mkdir -p $DST/$NAME

rsync -avz $SRC/$NAME/ui/img/* $HOST:$DST/$NAME/
