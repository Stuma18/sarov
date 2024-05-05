import streamlit as st


chart_visual = st.sidebar.radio('Содержание', 
    ('Главная', 'О SciPy', 'Для чего нужна SciPy', 'Отличия SciPy от NumPy', 'Возможности SciPy', 'Пакеты в SciPy', 'Особенности SciPy', 'Линейная алгебра и SciPy', 'Нахождение обратной матрицы', 'Нахождение определителей', 'Источники'))

if chart_visual == 'Главная':
    st.header('SciPy')
    from PIL import Image
    image = Image.open('Scipy.png')
    st.image(image)
    st.header('Студеникина Мария')
    st.header('Волкова Анна')
    st.header('Савкина Виктория')
  
if chart_visual == 'О SciPy':
    st.header('О SciPy')
    st.write('SciPy — это библиотека Python с открытым исходным кодом, предназначенная для решения научных и математических проблем.')
    st.write('SciPy написана в основном на Python и частично на языках C, C++ и Fortran.')

if chart_visual == 'Для чего нужна SciPy':
    st.header('Для чего нужна SciPy')
    st.markdown("""
    - Для сложных математических расчетов
    - Для проведения научных исследований
    - Для глубокого анализа данных
    - Для машинного обучения
    - Для формирования двумерных и трехмерных графиков
    """)

if chart_visual == 'Отличия SciPy от NumPy':
    st.header('Отличия SciPy от NumPy')
    st.write('В SciPy гораздо больше функций и методов, чем в NumPy')
    st.write('NumPy ориентирована на базовые вычисления и простую работу с матрицами, SciPy предназначена для глубокого научного анализа')
    st.write('NumPy не имеет дополнительных зависимостей, вместе с библиотекой не нужно ничего устанавливать. SciPy требует установки NumPy для корректной работы')

if chart_visual == 'Возможности SciPy':
    st.header('Возможности SciPy')
    st.markdown("""
    - Работа с продвинутыми математическими операциями
    - Обработка сигналов
    - Ввод и вывод файлов
    - Работа с изображениями и графиками
    """)

if chart_visual == 'Пакеты в SciPy':
    st.header('Пакеты в SciPy')
    if (st.button('cluster')):
        st.write('Алгоритмы кластерного анализа')
    
    if (st.button('constants')):
        st.write('Физические и математические константы')
    
    if (st.button('fftpack')):
        st.write('Быстрое преобразование Фурье')
    
    if (st.button('integrate')):
        st.write('Решения интегральных и обычных дифференциальных уравнений')
    
    if (st.button('interpolate')):
        st.write('Интерполяция и сглаживание сплайнов')
    
    if (st.button('io')):
        st.write('Ввод и вывод')

    if (st.button('linalg')):
        st.write('Линейная алгебра')
    
    if (st.button('ndimage')):
        st.write('N-размерная обработка изображений')
    
    if (st.button('odr')):
        st.write('Метод ортогональных расстояний')
    
    if (st.button('optimize')):
        st.write('Оптимизация и численное решение уравнений')

    if (st.button('signal')):
        st.write('Обработка сигналов')

    if (st.button('sparse')):
        st.write('Разреженные матрицы')

    if (st.button('spatial')):
        st.write('Разреженные структуры данных и алгоритмы')

    if (st.button('special')):
        st.write('Специальные функции')

    if (st.button('stats')):
        st.write('Статистические распределения и функции')

if chart_visual == 'Особенности SciPy':
    st.header('Особенности SciPy')
    st.markdown("""
    - Бесплатное распространение
    - Низкий порог входа
    - Быстрое исполнение кода
    """)

if chart_visual == 'Линейная алгебра и SciPy':
    st.header('Линейная алгебра и SciPy')
    st.write('SciPy обладает очень быстрыми возможностями линейной алгебры, поскольку он построен с использованием библиотек ATLAS LAPACK и BLAS.')

if chart_visual == 'Нахождение обратной матрицы':
    st.header('Нахождение обратной матрицы')
    code = '''import numpy as np
    from scipy import linalg
    A = np.array([[1,2],[4,3]])
    B = linalg.inv(A)
    print(B)'''
    st.code(code, language='python')
    
    if (st.button('Результат')):
        import numpy as np
        from scipy import linalg
        A = np.array([[1,2],[4,3]])
        B = linalg.inv(A)
        print(B)
        st.write(B)

if chart_visual == 'Нахождение определителей':
    st.header('Нахождение определителей')
    code = '''import numpy as np
    from scipy import linalg
    A = np.array([[1,2],[4,3]])
    B = linalg.det(A)
    print(B)'''
    st.code(code, language='python')

    if (st.button('Результат')):
        import numpy as np
        from scipy import linalg
        A = np.array([[1,2],[4,3]])
        B = linalg.det(A)
        print(B)
        st.write(B)

if chart_visual == 'Источники':
    st.header('Источники')
    st.write('П. Н. Вабищевич. Численные методы: Вычислительный практикум. — М.: Книжный дом  «ЛИБРОКОМ», 2010. — 320 с.')
    st.write('https://pythonim.ru/libraries/biblioteka-scipy-v-python?ysclid=l7yy4f15nl924001338')
    st.write('https://pythonru-com.turbopages.org/turbo/pythonru.com/s/biblioteki/scipy-python')
    st.write('https://blog-skillfactory-ru.turbopages.org/turbo/blog.skillfactory.ru/s/glossary/scipy/')
