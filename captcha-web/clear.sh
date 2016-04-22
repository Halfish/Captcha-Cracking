kill -9 `lsof -i:3000 | grep python | awk -F ' ' '{print $2}' | uniq` 2> /dev/null
