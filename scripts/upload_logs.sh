#!/bin/bash

if [ $# -ne 1 ]
then
	echo "illegal number of params. help: $0 <path-to-logdir>"
	exit -1
fi

token=`cat $(dirname $0)/.token`
psw=`cat $(dirname $0)/.password`
logdir=$1

echo -e "[Info] Creating tar ball for log dir ${logdir}\n"

new_basename=$(basename -- $logdir)
for s in "{" "}"
do
	new_basename=$(echo $new_basename | sed "s/${s}//g")
done

tarball="${USER}_$(date '+%d%m%Y_%H%M%S')_${new_basename}.tar"

tar cvf ${tarball} ${logdir}
echo -e "[Info] Created ${tarball}\n"

echo "[Info] Uploading ${tarball} ..."

cmd="time curl -u ${token}:${psw} -T ${tarball} \"https://owncloud.tuwien.ac.at/public.php/webdav/${tarball}\""
echo $cmd
eval $cmd

echo "[Info] Done."
