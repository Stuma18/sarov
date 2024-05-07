#!/bin/bash

# Чтение файла в массив построчно
getArray() {
    array=() # Create array
    while IFS= read -r line # чтение чтроки
    do
        array+=("$line") # добавление с
    done < "$1"
}
getArray "log_time.txt"


name=()
# Оставляем часть массива между двумя пробелами
for element in "${array[@]}"; do
    result=$(echo "$element" | awk '{print $2}')
    name+=("$result")
    #echo "$result"
done

#for element in "${name[@]}"; do
#    echo "$element"
#done

unique_array=($(echo "${name[@]}" | tr ' ' '\n' | sort -u))

#echo "${unique_array[@]}"

# Подсчет длины массива
length=${#unique_array[@]}

#echo "Длина массива: $length"


printf "%-40s %-20s %-20s\n" "USERNAME " "SUM_TIME" "TOTAL_JOBS "
printf "%-40s\n" "AVG_RUN "
echo "-----------------------------------------------------------------"
echo "--------------------------------"

for string in "${unique_array[@]}"
do
    sum_time=0
    index=0
    for line in "${array[@]}"
    do
        if [[ $line == *$string* ]]; then
            if [[ $line  =~ $string[[:space:]]+([0-9]+) ]]; then
                number="${BASH_REMATCH[1]}"
                #echo "Число: $number"
                sum_time=$((sum_time + number)) 
                index=$((index+1))
            fi
        fi
    
    done

    sred=$((sum_time / index))
    #echo "SUM_TIME: $sum_time"
    #echo "TOTAL_JOBS: $index"
    #echo "AVG_RUN: $sred"

    
    printf "%-40s %-20s %-20s\n" "$string" "$sum_time" "$index"
    printf "%-40s\n" "$sred"


done

