#!/bin/bash
branch=$(git branch | grep "*" | cut -f 2 -d '*')
echo "${branch//[[:space:]]/}"
