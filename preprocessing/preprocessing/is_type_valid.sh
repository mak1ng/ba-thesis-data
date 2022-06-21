#!/bin/bash
while getopts m:c: flag
do
    case "${flag}" in
        m) module=${OPTARG};;
        c) class=${OPTARG};;
    esac
done
#activate project env
source $HOME/deeptyper_preprocessing/bin/activate

function is_valid {
    python3 - <<END
import importlib as imp
try:
    if "$module" == "" and "$class" == "None":
        print("None type")
    elif "$module" == "":
        print("No module path")
    else:
        getattr(imp.import_module("$module"), "$class")
except:
    print("invalid")
END
}
is_valid
