#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
while getopts n:p: flag
do
    case "${flag}" in
        n) predictions=${OPTARG};;
       # p) projetcs_directory=${OPTARG};;
    esac
done

projects_directory=$SCRIPT_DIR
#conda install openmpi
#apt install git
#pip install virtualenv
source $HOME/conda/etc/profile.d/conda.sh
#conda activate deeptyper_3.6
#conda install openmpi
#pip install cntk
#pip install requests
#pip install libcst
#pip install pygments
#export LD_LIBRARY_PATH=$HOME/../opt/conda/envs/deeptyper_3.6/lib/ 
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/../opt/conda/pkgs/openmpi-4.0.2-hb1b8bf9_1/lib 
#ln -s $HOME/../opt/conda/lib/libmpi.so.1 $HOME/../opt/conda/envs/deeptyper_3.6/lib/libmpi.so.12
#ln -s $HOME/../opt/conda/pkgs/openmpi-4.0.2-hb1b8bf9_1/lib/libmpi_cxx.so.40 $HOME/../opt/conda/envs/deeptyper_3.6/lib/libmpi_cxx.so.1
#conda deactivate
mkdir $projetcs_directory/projects

#get name, repository link and version tag for each project in json-file
readarray -t project_names < <(jq -c '.projects[].name' $SCRIPT_DIR/projects_new_version.json)
echo "________________________Total number of projects: ${#project_names[@]}________________________"
echo "Filepath, Skipped Types, AnnTotal, #Annotations, #SkippedTypes, #NoneSkips, #InvalidSkips, #MissingPath" >> $SCRIPT_DIR/results_invalid.csv
#count total number of modules
a=0
for project_name in "${project_names[@]}"
do 
   #get all modules according to project name
   readarray -t project_modules < <(jq --arg p_name ${project_name:1: -1} '.projects[] | select(.name==$p_name) | .modules[]' $SCRIPT_DIR/projects_new_version.json)
   #get repository link
   repo=$(jq --arg p_name ${project_name:1: -1} '.projects[] | select(.name==$p_name) | .repository' $SCRIPT_DIR/projects_new_version.json)
   #get version tag
   tag=$(jq --arg p_name ${project_name:1: -1} '.projects[] | select(.name==$p_name) | .version' $SCRIPT_DIR/projects_new_version.json)
   #get src projetcs_directory
   src_dir=$(jq --arg p_name ${project_name:1: -1} '.projects[] | select(.name==$p_name) | .sources' $SCRIPT_DIR/projects_new_version.json)
   
   cd $projetcs_directory/projects
   git clone --depth 1 --branch ${tag:1: -1} ${repo:1: -1}
   cd $HOME/.. 
   
   #create virtual environment
   cd
   virtualenv --python=$HOME/conda/bin/python3.9 $HOME/deeptyper_preprocessing
   source $HOME/deeptyper_preprocessing/bin/activate
   #install according project package
   pip install git+${repo:1: -1}@${tag:1: -1}
   deactivate
   
   declare -a paths
   paths=()
   for py_file in "${project_modules[@]}"
   do
      #create file path
      path="$projetcs_directory/${src_dir:1: -1}/${py_file:1: -1}"
      path=${path//.//}.py
      paths+=("$(echo $path)")
      #activate environment for deeptyper
      #eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
      conda activate deeptyper_3.6
      
      #TODO: abaendern
      #export LD_LIBRARY_PATH=$HOME/../opt/conda/envs/deeptyper_3.6/lib/ 
      #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/../opt/conda/pkgs/openmpi-4.0.2-hb1b8bf9_1/lib
      cd $HOME/preprocessing/
      #start preprocessing for module
      python3 $SCRIPT_DIR/preprocess_modules_skip_env.py -m $path -n $predictions
   done
   echo "________________________Project $project_name had ${#project_modules[@]} modules.________________________"
   a=$(($a+${#project_modules[@]}))
   conda deactivate
   source $HOME/deeptyper_preprocessing/bin/activate
   #install modified version of autoimport to add missing import statements
   pip install $SCRIPT_DIR/autoimport-1.2.2_modified
   #fix missing import statements for files including predictions
   for f_path in "${paths[@]}"
   do
      for ((i=0;i<$predictions;i++))
      do 
         echo "Fix Imports for:"${f_path:0: -3}_$i.py
         autoimport ${f_path:0: -3}_$i.py
      done
   done
   #remove environment
   rm -r $HOME/deeptyper_preprocessing
done
echo "Total number of modules:" $a
