#!/bin/bash

while true; do
    echo "Введите выражение (введите 'exit' для выхода):"
    read expression

    if [ "$expression" == "exit" ]; then
        break
    fi

    # Создаем заготовку Си-файла с вставленным выражением
    cat <<EOF > calc.c
#include <stdio.h>
#include <math.h>

int main() {
   int result = $expression;
   printf("%d\n", result);
   return 0;
}
EOF

    # Компилируем Си-файл
    gcc -o calc calc.c -lm

    # Запускаем исполняемый файл
    ./calc
done
