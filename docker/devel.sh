#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--img)
    IMAGE="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ -z "$IMAGE" -a "$IMAGE" != " " ]; then
    IMAGE=nodeflux/vortex
fi

docker run --runtime=nvidia -it ${POSITIONAL[@]} -v $DIR/../:/app/ $IMAGE bash