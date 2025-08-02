# file:  webserver.sh
previous=`ps aux | grep python3 |grep http.server| awk '{print $2}'`
if [[ $previous > "" ]];then 
  kill -9 $previous
fi
python3 -m http.server --bind localhost
