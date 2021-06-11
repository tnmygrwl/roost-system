#!/bin/bash

NAME=${1:-test}
SRC=${2:-/mnt/nfs/scratch1/wenlongzhao/roosts_data}
HOST=${3:-doppler.cs.umass.edu}
DST=${4:-/var/www/html/roost/img}

ssh $HOST mkdir -p $DST/$NAME

# change after source folder structure is updated
rsync -avz $SRC/$NAME/roosts_ui_data/ref0.5_images/ $HOST:$DST/$NAME/dz05
rsync -avz $SRC/$NAME/roosts_ui_data/rv0.5_images/ $HOST:$DST/$NAME/vr05

