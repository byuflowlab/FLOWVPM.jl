#!/bin/bash

# Variables to replace
v=p
# v=P
# v=source_particle

# Replacements
sed -i -e "s/$v\[1\]/get_X($v)[1]/g" $1
sed -i -e "s/$v\[2\]/get_X($v)[2]/g" $1
sed -i -e "s/$v\[3\]/get_X($v)[3]/g" $1

sed -i -e "s/$v\[4\]/get_Gamma($v)[1]/g" $1
sed -i -e "s/$v\[5\]/get_Gamma($v)[2]/g" $1
sed -i -e "s/$v\[6\]/get_Gamma($v)[3]/g" $1

sed -i -e "s/$v\[7\]/get_sigma($v)[]/g" $1
sed -i -e "s/$v\[8\]/get_vol($v)[]/g" $1
sed -i -e "s/$v\[9\]/get_circulation($v)[]/g" $1

sed -i -e "s/$v\[10\]/get_U($v)[1]/g" $1
sed -i -e "s/$v\[11\]/get_U($v)[2]/g" $1
sed -i -e "s/$v\[12\]/get_U($v)[3]/g" $1

sed -i -e "s/$v\[16\]/get_J($v)[1]/g" $1
sed -i -e "s/$v\[17\]/get_J($v)[2]/g" $1
sed -i -e "s/$v\[18\]/get_J($v)[3]/g" $1
sed -i -e "s/$v\[19\]/get_J($v)[4]/g" $1
sed -i -e "s/$v\[20\]/get_J($v)[5]/g" $1
sed -i -e "s/$v\[21\]/get_J($v)[6]/g" $1
sed -i -e "s/$v\[22\]/get_J($v)[7]/g" $1
sed -i -e "s/$v\[23\]/get_J($v)[8]/g" $1
sed -i -e "s/$v\[24\]/get_J($v)[9]/g" $1

sed -i -e "s/$v\[25\]/get_PSE($v)[1]/g" $1
sed -i -e "s/$v\[26\]/get_PSE($v)[2]/g" $1
sed -i -e "s/$v\[27\]/get_PSE($v)[3]/g" $1

sed -i -e "s/$v\[28\]/get_M($v)[1]/g" $1
sed -i -e "s/$v\[29\]/get_M($v)[2]/g" $1
sed -i -e "s/$v\[30\]/get_M($v)[3]/g" $1
sed -i -e "s/$v\[31\]/get_M($v)[4]/g" $1
sed -i -e "s/$v\[32\]/get_M($v)[5]/g" $1
sed -i -e "s/$v\[33\]/get_M($v)[6]/g" $1
sed -i -e "s/$v\[34\]/get_M($v)[7]/g" $1
sed -i -e "s/$v\[35\]/get_M($v)[8]/g" $1
sed -i -e "s/$v\[36\]/get_M($v)[9]/g" $1

sed -i -e "s/$v\[37\]/get_C($v)[1]/g" $1
sed -i -e "s/$v\[38\]/get_C($v)[2]/g" $1
sed -i -e "s/$v\[39\]/get_C($v)[3]/g" $1

sed -i -e "s/$v\[40\]/get_S($v)[1]/g" $1
sed -i -e "s/$v\[41\]/get_S($v)[2]/g" $1
sed -i -e "s/$v\[42\]/get_S($v)[3]/g" $1
