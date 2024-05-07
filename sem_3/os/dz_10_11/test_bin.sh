#!/bin/bash

# Определение пути к исполняемому файлу калькулятора
calculator="./calc"

# Определение пути к файлу с эталонными результатами
expected_results="./expected_results.txt"

# Определяем функцию для сравнения двух строк
function compare_results {
    if [ "$1" == "$2" ]; then
        echo "Результат: ПРОЙДЕН"
    else
        echo "Результат: НЕ ПРОЙДЕН"
    fi
}

# Тест 1: сложение
echo "Тест 1: сложение"
result=$(echo "2+2" | sh ./bin.sh)

#result=$(echo "2+2" | $calculator)
expected_result="4"
compare_results "$result" "$expected_result"

# Тест 2: вычитание
echo "Тест 2: вычитание"
result=$(echo "5-3" | sh ./bin.sh)
expected_result="2"
compare_results "$result" "$expected_result"

# Тест 3: умножение
echo "Тест 3: умножение"
result=$(echo "4*3" | sh ./bin.sh)
expected_result="12"
compare_results "$result" "$expected_result"

# Тест 4: деление
echo "Тест 4: деление"
result=$(echo "10/5" | sh ./bin.sh)
expected_result="2"
compare_results "$result" "$expected_result"

# Тест 5: возведение в степень
echo "Тест 5: возведение в степень"
result=$(echo "pow (2, 3)" | sh ./bin.sh)
expected_result="8"
compare_results "$result" "$expected_result"

# Тест 6: вычисление выражения с использованием скобок
echo "Тест 6: вычисление выражения с использованием скобок"
result=$(echo "2*(3+4)" | sh ./bin.sh)
expected_result="14"
compare_results "$result" "$expected_result"

