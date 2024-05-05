#!/bin/bash

#cd keys2 
echo "Введите название папки:"
read folder_name
cd $folder_name

directory="$PWD"

echo " "
# проверяет каждый ключ на правильность расширения и прав файла
# если необходимо, исправляет их
for file in "$directory"/*; 
do
    if [ -f "$file" ]; then
        extension="${file##*.}"
        filename="${file%.*}"

        if [ "$extension" != "pub" ] || [ ! -e "$file" ]; then
            mv "$file" "$filename.pub"
            #echo "Renamed $file to $filename.pub"
        fi

        if [ -x "$file" ]; then
            chmod -x "$file"
            #echo "Removed execute permissions from $file"
        fi
    fi
done

# корректность ключа
# выводим сообщения такой-то ключ конвертировался, 
# если конвертация не удалась, то выдать сообщение, что пользователь с таким-то логином прислал некорректный ключ
for file in "$directory"/*; 
do
  if [ -f "$file" ]; then
        ssh-keygen -l -f "$file" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            ssh-keygen -i -f "$file" > temp_file 2>&1
            if [ $? -eq 0 ]; then
                cp temp_file "$file"
                echo "Ключ исправлен у пользователя:"
                #echo "$file" | cut -d '2' -f2- | cut -d '/' -f2- | cut -d '/' -f2- | cut -d'.' -f1
                echo "$file" | tail -c 29 | cut -d'.' -f1
                echo " "
            else
                echo "Не коректный ключ у пользователя (пользователь полный м...):"
                #echo "$file" | cut -d '2' -f2- | cut -d '/' -f2- | cut -d '/' -f2- | cut -d'.' -f1
                echo "$file" | tail -c 29 | cut -d'.' -f1
                echo " "
                #echo "$file"
                rm "$file"
            fi
            rm temp_file
        fi
    fi
done

# создаем папку и переносим файлы
# Проходим по всем файлам в директории
for file in *
do
    # Получаем первые 22 символа в названии файла
    prefix=$(echo $file | cut -c1-21)
    
    # Создаем папку с таким же названием, если ее еще нет
    if [ ! -d "$prefix" ]; then
        mkdir "$prefix"
    fi
    
    # Перемещаем файл в созданную папку
    mv "$file" "$prefix"
done

echo "Файлы успешно перемещены в папки:"
ls "$directory"
echo " "



