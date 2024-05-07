#!/bin/bash

#echo "Введите выражение:"
read expression

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

# Удаляем временные файлы
rm calc.c calc

# sh bin.sh