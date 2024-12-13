#!/usr/bin/env bash
if [ $(git rev-parse --abbrev-ref HEAD) == "master" ]; then
    # make validate
    echo "Don't forget to call 'make validate'"
fi
