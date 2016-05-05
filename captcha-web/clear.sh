kill -9 `lsof -i:3000 | grep python | awk -F ' ' '{print $2}' | uniq` 2> /dev/null
rm alpha.png beta.png gamma.png a.txt binary.jpg 2> /dev/null
