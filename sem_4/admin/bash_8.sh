#!/bin/bash

# Генерируем случайные операнды, если не были переданы через командную строку
if [ -z "$1" ] || [ -z "$2" ]; then
    num1=$((RANDOM % 9000 + 1000))
    num2=$((RANDOM % 9 + 1))
    while true; 
    do
    if (( $num1 % $num2 != 0 )); then
        break
    else
        num2=$((RANDOM % 9 + 1))
    fi
done
else
    num1=$1
    num2=$2
fi

# Очистить экран
clear

# Выводим пример деления в середине терминала
tput sc # Сохранить текущее положение курсора
tput cup $(($(tput lines) / 2)) $(($(tput cols) / 2 - 2))
echo "$num1" "|" "$num2" 
tput cup $(($(tput lines) / 2 + 1)) $(($(tput cols) / 2 + 4))
echo "----"
tput rc # Восстановить положение курсора

# Выполняем деление в столбик
echo ""
echo "Enter the result step by step:"
echo ""

result1=$((num1/num2))
result2=$((num1%num2))

echo $result1
index=0

len_1=${#result1}
tp=0

while [ $index -lt $len_1 ];
do
    tput cup $(($(tput lines) / 2 + 2)) $(($(tput cols) / 2 + 4 + index))
    read user_number
    if [ $user_number -eq ${result1:index:1} ]; then
        echo "Yes! You are right! "
    else
        echo "No! You are wrong! Try again! "
        continue
    fi

    if [ $user_number -ne 0 ]; then
        tp=$((tp+2))
        num3=$((user_number*num2))
        len_3=${#num3}
        nul=${result1:index:1}
        if [ $nul -ne 0 ]; then
            if [ $index -eq 0 ]; then
                tap=1
            else
                tap=$((tap+2))
            fi
        fi

        if [ $index -ne 0 ]; then
            if [ $len_3 -ne 1 ]; then
                tput rc
                #tput cup $(($(tput lines) / 2 + index + index + 1)) $(($(tput cols) / 2 - 2 + index - 1))
                tput cup $(($(tput lines) / 2 + tap)) $(($(tput cols) / 2 - 2 + index - 1))
                read user_number_2
            else
                tput rc
                #tput cup $(($(tput lines) / 2 + index + index + 1)) $(($(tput cols) / 2 - 2 + index))
                tput cup $(($(tput lines) / 2 + tap)) $(($(tput cols) / 2 - 2 + index))
                read user_number_2
            fi
            else
                if [ $len_3 -ne 1 ]; then
                    tput rc
                    #tput cup $(($(tput lines) / 2 + index + index + 1)) $(($(tput cols) / 2 - 2 + index - 1))
                    tput cup $(($(tput lines) / 2 + tap)) $(($(tput cols) / 2 - 2 + index))
                    read user_number_2
                else
                    tput rc
                    #tput cup $(($(tput lines) / 2 + index + index + 1)) $(($(tput cols) / 2 - 2 + index))
                    tput cup $(($(tput lines) / 2 + tap)) $(($(tput cols) / 2 - 2 + index))
                    read user_number_2
                fi
        fi

        minus1=0

        if [ $user_number_2 -eq $num3 ]; then
            echo "Yes! You are right! "
        else
            echo "No! You are wrong! Try again! "
            continue
        fi
        
        if [ $index -eq 0 ]; then
            if [ $len_3 -eq 1 ]; then
                first=${num1:index:1}
            else
                first=${num1:index:2}
            fi
        fi
        #else

        
        #echo $first
        minus1=$((first-num3))
        
        #echo $minus1
        if [ $minus1 -ge 0 ]; then
            if [ $minus1 -ne 0 ]; then
                second=${num1:index+1:1}
                tput cup $(($(tput lines) / 2 + index + index + 2)) $(($(tput cols) / 2 - 2 + index))
                print="$minus1$second"
                echo $print
            else
                second=${num1:index+1:1}
                if [ $second -ne 0 ]; then
                    if [ $second -ge $num2 ]; then
                        second=${num1:index+1:1}
                        tput cup $(($(tput lines) / 2 + index + index + 2)) $(($(tput cols) / 2 - 2 + index + 1))
                        echo $second
                    else
                        second=${num1:index+1:2}
                        tput cup $(($(tput lines) / 2 + index + index + 2)) $(($(tput cols) / 2 - 2 + index + 1))
                        echo $second
                    fi
                fi
            fi
        else
            second=${num1:index+1:1}
            if [ $second -lt $num2 ]; then
                second=${num1:index+1:2}
            fi
            tput cup $(($(tput lines) / 2 + index + index + 2)) $(($(tput cols) / 2 - 2 + index + 1))
            echo $second
        fi
    fi

    index=$((index+1))
done


tput rc
tput cup $(($(tput lines) / 2 + tp)) $(($(tput cols) / 2 - 1))
echo '----'
tput cup $(($(tput lines) / 2 + tp + 1)) $(($(tput cols) / 2 - 3 + len_1))
echo $result2
