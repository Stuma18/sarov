import streamlit as st

st.set_page_config(page_title = "Эпсилон-жадный алгоритм")

st.sidebar.success("Эпсилон-жадный алгоритм")

#st.sidebar.button("Описание алгоритма")
#if st.sidebar.button('Моделирование работы алгоритма'):

        
status = st.sidebar.radio("Выберете: ", ("Описание эпсилон-жадного алгоритма", 'Моделирование работы эпсилон-жадного алгоритма'))
if (status == 'Моделирование работы эпсилон-жадного алгоритма'):
        
        import streamlit as st
        import numpy as np 
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import random
        from prettytable import PrettyTable
        from toolz import partition



        st.write("# Моделирование работы эпсилон-жадного алгоритма")
        
        #eps = st.slider('Эпсилон:', 0.1, 0.2, 0.1)

        #N_takt = st.slider('Количество тактов:', 10, 2000, 100, step = 10)

        eps = st.sidebar.slider('Эпсилон:', 0.1, 0.2, 0.1)

        N_takt = st.sidebar.slider('Количество тактов:', 10, 400, 100, step = 10)
        
        st.sidebar.write("\n")
        st.sidebar.write("\n")
        st.sidebar.write("\n")
        st.sidebar.write("\n")

        #c_0 = st.sidebar.slider('Конверсия баннера 0:', 0.001, 0.10, 0.010, step = 0.001)

        #c_1 = st.sidebar.slider('Конверсия баннера 1:', 0.010, 0.10, 0.012, step = 0.001)

        #c_2 = st.sidebar.slider('Конверсия баннера 2:', 0.010, 0.100, 0.014, step = 0.001)

        #conversion = [c_0, c_1, c_2]

        # задаем эпсилон и количество тактов
        #eps = 0.1
        #N_takt = 1000

        # количество пользователей
        use_0 = 0
        use_1 = 0
        use_2 = 0

        # конверсия
        conversion = [1.2*0.98/100, 1.2/100, 1.2*0.97/100]
        
        #  сайт 1- (на 2% меньше, чем 2-й сайт)
        #  сайт 2 - =1.2%
        #  сайт 3 - (на 3% меньше, чем 2-й сайт)

        # количество конверсий
        n = 0
        n_0 = 0
        n_1 = 0
        n_2 = 0

        # оценка вероятности конверсии
        estimate_0 = 0
        estimate_1 = 0
        estimate_2 = 0


        mytable_2 = PrettyTable()
        mytable_2.field_names = ["Номер такта", "Номер банера", "Количество пользователей", "Количество конверсий", "Оценка вероятности конверсии"]


        # показываем баннер и меняем его показатели
        def p(n, Con, use):
        # задаем количество пользователей
                N_users = 10000
                Users_tab = 1000
                
                while N_users > 0:
                        
                        # рекламный банер
                        P = random.random()
                
                        if P < Con:
                                n += 1
                        
                        N_users -= 1
                
                use += Users_tab
                estimate = n / use
                
                return estimate, n, use


        # поиск максимального
        def min_max(banners):
                if banners[0]['estimate'] > banners[1]['estimate']:
                        if banners[0]['estimate'] > banners[2]['estimate']:
                                return 0
                        else:
                                return 2
                        
                else:
                        if banners[1]['estimate'] > banners[2]['estimate']:
                                return 1
                        else:
                                return 2
        

        # поиск 2-ух минимальных
        def min_max_2(banners):
                if ban_0['estimate'] > banners[1]['estimate']:
                        if ban_0['estimate'] > banners[2]['estimate']:
                                return 1, 2
                        else:
                                return 0, 1
                
                else:
                        if banners[1]['estimate'] > banners[2]['estimate']:
                                return 0, 2
                        else:
                                return 0, 1

        def get_row(st, ban):
                return [st, ban['num'], ban['n_users'], ban['n_conv'], ban['estimate']]

        fig, ax = plt.subplots()
        ax.set(xlabel='Такты', 
        ylabel='Конверсия', 
        title='Моделирование работы эпсилон-жадного алгоритма')
        colors = ["red",
                "black",
                "magenta",
                "blue",
                "grey",
                ]
        blue_patch = mpatches.Patch(color = 'blue', label = 'Баннер 0')
        green_patch = mpatches.Patch(color = 'green', label = 'Баннер 1')
        magenta_patch = mpatches.Patch(color = 'magenta', label = 'Баннер 2')
        ax.legend(handles = [blue_patch, green_patch, magenta_patch])

        x = np.arange(N_takt)

        estimate_0, n_0, use_0 = p(n_0, conversion[0], use_0)
        ban_0 = {'num' : '0', 'n_users' : use_0, 'n_conv' : n_0, 'estimate' : estimate_0}
        mytable_2.add_row(get_row("Такт 0", ban_0))
        ax.plot(0, ban_0['estimate'], "bo")

        estimate_1, n_1, use_1 = p(n_1, conversion[1], use_1)
        ban_1 = {'num' : '1', 'n_users' : use_1, 'n_conv' : n_1, 'estimate' : estimate_1}
        mytable_2.add_row(get_row("Такт 1", ban_1))
        ax.plot(1, ban_1['estimate'], "go")

        estimate_2, n_2, use_2 = p(n_2, conversion[2], use_2)
        ban_2 = {'num' : '2', 'n_users' : use_2, 'n_conv' : n_2, 'estimate' : estimate_2}
        mytable_2.add_row(get_row("Такт 2", ban_2))
        ax.plot(2, ban_2['estimate'], "mo")

        banners = [ban_0, ban_1, ban_2]


        # массив для построения графика 1
        grafik = []
        grafik.append(ban_0['estimate'])
        grafik.append(ban_1['estimate'])
        grafik.append(ban_2['estimate'])


        # массив для построения графика 2
        grafik_2 = []
        grafik_2.append(ban_0['num'])
        grafik_2.append(ban_1['num'])
        grafik_2.append(ban_2['num'])


        i = 3
        # показываем баннеры
        while i < N_takt:
                rand = random.random()
                if rand > eps:
                        max_ind = min_max(banners)

                        estimate_0, n_0, use_0 = p(banners[max_ind]['n_conv'], conversion[max_ind], banners[max_ind]['n_users'])
                        banners[max_ind]['n_users'] = use_0
                        banners[max_ind]['n_conv'] = n_0
                        banners[max_ind]['estimate'] = estimate_0

                        str_0 = "Такт " + str(i)
                        mytable_2.add_row(get_row(str_0, banners[max_ind]))

                        # точки для графика 1
                        grafik.append(banners[max_ind]['estimate'])
                        if int(banners[max_ind]['num']) == 0:
                                ax.plot(i, banners[max_ind]['estimate'], "bo")
                        if int(banners[max_ind]['num']) == 1:
                                ax.plot(i, banners[max_ind]['estimate'], "go")
                        if int(banners[max_ind]['num']) == 2:
                                ax.plot(i, banners[max_ind]['estimate'], "mo")
                        
                        # массив для построения графика 2   
                        grafik_2.append(banners[max_ind]['num'])
                        
                        i += 1
                        
                else:
                        mid_ind, min_ind = min_max_2(banners)

                        estimate_0, n_0, use_0 = p(banners[mid_ind]['n_conv'], conversion[mid_ind], banners[mid_ind]['n_users'])
                        banners[mid_ind]['n_users'] = use_0
                        banners[mid_ind]['n_conv'] = n_0
                        banners[mid_ind]['estimate'] = estimate_0

                        str_1 = "Такт " + str(i)
                        mytable_2.add_row(get_row(str_1, banners[mid_ind]))

                        # точки для графика 1
                        grafik.append(banners[mid_ind]['estimate'])
                        if int(banners[mid_ind]['num']) == 0:
                                ax.plot(i, banners[mid_ind]['estimate'], "bo")
                        if int(banners[mid_ind]['num']) == 1:
                                ax.plot(i, banners[mid_ind]['estimate'], "go")
                        if int(banners[mid_ind]['num']) == 2:
                                ax.plot(i, banners[mid_ind]['estimate'], "mo")
                        
                        # массив для построения графика 2   
                        grafik_2.append(banners[mid_ind]['num'])
                        
                        i += 1

                        estimate_0, n_0, use_0 = p(banners[min_ind]['n_conv'], conversion[min_ind], banners[min_ind]['n_users'])
                        banners[min_ind]['n_users'] = use_0
                        banners[min_ind]['n_conv'] = n_0
                        banners[min_ind]['estimate'] = estimate_0
                        
                        if i < N_takt:
                                str_2 = "Такт " + str(i)
                        mytable_2.add_row(get_row(str_2, banners[min_ind]))
                        
                        # точки для графика 1
                        grafik.append(banners[min_ind]['estimate'])
                        if int(banners[min_ind]['num']) == 0:
                                ax.plot(i, banners[min_ind]['estimate'], "bo")
                        if int(banners[min_ind]['num']) == 1:
                                ax.plot(i, banners[min_ind]['estimate'], "go")
                        if int(banners[min_ind]['num']) == 2:
                                ax.plot(i, banners[min_ind]['estimate'], "mo")
                        
                        # массив для построения графика 2   
                        grafik_2.append(banners[min_ind]['num'])
                        
                        i += 1

        print(mytable_2) 

        # для графика 1
        grafik_new = grafik[:N_takt]
        ax.plot(x, grafik_new, c = "grey", linestyle = 'solid', lw = 1.5)
        if N_takt <= 20:
                ax.set_xticks(np.arange(0, N_takt, 1))
        ax.grid(True)


        # для графика 2
        fig_2, ax_2 = plt.subplots()

        ax_2.set(xlabel='Такты', 
                ylabel='Вероятность показа баннера $2$', 
                title='Частота показа баннера $2$')
        colors = ["red",
                "black",
                "magenta",
                "blue",
                "grey",
                ]

        n_mas = 0           # номер новых массивов
        col_takt = 10       # частота расчетов
        ver = []            # массив вероятности показа баннера 2 в новых массивах
        massiv = list(partition(col_takt, grafik_2))
        while n_mas < N_takt / col_takt:
                massiv_2 = massiv[n_mas]
                a = 0               # номер числа в массиве
                b = 0               # количество показов баннера 2 в новых массивах
                while a < col_takt:
                        if int(massiv_2[a]) == 2:
                                b += 1
                        a += 1
                ver_0 = b / col_takt
                ver.append(ver_0)
                
                n_mas += 1

        x_2 = np.arange(N_takt / col_takt)* 10
        ax_2.plot(x_2, ver, c = "grey", linestyle = 'solid', lw = 1.5)
        ax_2.grid(True)
        ax_2.set_ylim(-0.1, 1.1)

        #plt.show()

        st.pyplot(fig)
        #st.pyplot(fig_2)

        #st.button("Re-run")

        #st.write(mytable_2)










else:

        st.write("# Эпсилон-жадный алгоритм")

        st.markdown(
        """
Эпсилон-жадный алгоритм — это один из наиболее популярных алгоритмов 
для решения дилеммы исследования-использования в машинном обучении.

Алгоритм является простым и интуитивно понятным. Он делает случайный выбор между исследованием и использованием. 
С вероятностью ε, называемой коэффициентом эксплорации, алгоритм случайным образом выбирает действие для исследования. 
В остальных случаях, алгоритм выбирает действие, которое имеет максимальную стоимость на основе текущих оценок.

""")


        st.markdown(
        """
## Работа эпсилон-жадного алгоритма
""")

        from PIL import Image
        image = Image.open('4.png')
        st.image(image)