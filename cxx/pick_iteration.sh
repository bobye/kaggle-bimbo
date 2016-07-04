#!/bin/bash
cat $1 | sed 's/\[\([0-9]*\)\].*:\([0-9\.]*\).*:\(\)/\1 \2 /' | grep "^[0-9]" | awk '{print $1,2*$3-$2}' | awk 'NR == 1 || $2 < min {line = $0; min = $2}END{print line}'
