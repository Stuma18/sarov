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
#tput sc # Сохранить текущее положение курсора
tput cup $(($(tput lines) / 2)) $(($(tput cols) / 2 - 2))
echo "$num1" "|" "$num2" 
tput cup $(($(tput lines) / 2 + 1)) $(($(tput cols) / 2 + 4))
echo "----"
#tput rc # Восстановить положение курсора

x_divisor=$(($(tput cols) / 2 + 4))
y_divisor=$(($(tput lines) / 2 + 2))

y_num=$(($(tput lines) / 2 + 1))
x_num=$(($(tput cols) / 2 - 2))
# Выполняем деление в столбик
echo ""
echo "Enter the result step by step:"
echo ""

result1=$((num1/num2))
result2=$((num1%num2))

echo $result1

cnum=0
ndigits=0
while [ $cnum -lt $num2 ]; do
    ndigits=$(($ndigits + 1))
    cnum=${num1:idx:ndigits}
done

idx=${#cnum}

while [ 1 ]; do
    divisor=$((cnum / num2))
    rem=$((cnum % num2))
    tput cup $y_divisor $x_divisor
    read -n1 useans
    while [ $useans != $divisor ]; do
        printf "\b \b"
        #notcorect
        #back to cup
        read -n1 useans
    done
    x_divisor=$((x_divisor + 1))

    mult=$((divisor * num2))
    indent=$((${#cnum} - ${#mult}))
    #x_num=$((x_num + indent))
    tput cup $y_num $x_num
    for (( i = 0; i < ${#mult}; i++)); do
        tput cup $y_num $((x_num + indent + i)) 
        # $((x_num + i))
        read -n1 c

        while [ $c != ${mult:i:1} ]; do
            printf "\b \b"
            #notcorect
            #back to cup
            read -n1 c
        done
    done

    y_num=$((y_num + 1))
    sub=$((cnum - mult))
    #if [ $sub != 0 ]; then
        indent=$((${#cnum} - ${#sub}))
        tput cup $y_num $((x_num + indent))
        read -n1 c
        while [ $c != $sub ]; do
            printf "\b \b"
            #notcorect
            #back to cup
            read -n1 c
        done
    #fi

    x_num1=$((x_num + ${#cnum}))
    x_num=$((x_num + ${#cnum} - ${#sub}))
    #idx=$((idx + ${#cnum}))

    cnum=$sub

    #counter=0
    for (( ; idx < ${#num1}; idx++)); do

        

        tput cup $y_num $x_num1
        read -n1 c

        while [ $c != ${num1:idx:1} ]; do
            printf "\b \b"
            #notcorect
            #back to cup
            read -n1 c
        done

        if [ $cnum == 0 -a $c != 0 ]; then
            x_num=$x_num1
        fi

        cnum=$((cnum*10+c))
        if [ $cnum -ge $num2 ]; then
            idx=$((idx + 1))
            break
        else
            tput cup $y_divisor $x_divisor
            read -n1 useans
            while [ $useans != 0 ]; do
                printf "\b \b"
                #notcorect
                #back to cup
                read -n1 useans
            done
            x_divisor=$((x_divisor + 1))
        fi
        x_num1=$((x_num1 + 1))
        #counter=$((counter + 1))
    done
   
    # ostatok
    if [ $cnum -lt $num2 ]; then
        break
    fi
    y_num=$((y_num + 1))
done

tput cup $((y_num + 1)) 0
