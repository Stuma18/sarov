import streamlit as st

st.set_page_config(page_title = "Алгоритм верхнего доверительного интервала")

st.sidebar.success("Алгоритм верхнего доверительного интервала")

#st.sidebar.button("Описание алгоритма")
#if st.sidebar.button('Моделирование работы алгоритма'):

status = st.sidebar.radio("Выберете: ", ("Описание алгоритма верхнего доверительного интервала", 'Моделирование работы алгоритма верхнего доверительного интервала'))
if (status == 'Моделирование работы алгоритма верхнего доверительного интервала'):
        
        st.write("# Моделирование работы алгоритма верхнего доверительного интервала")
        
        import streamlit as st
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches


        #st.sidebar.success("Моделирование работы алгоритма верхнего доверительного интервала")

        n_hours = st.sidebar.slider('Количество тактов:', 10, 400, 100, step = 10)

        st.sidebar.write("\n")
        st.sidebar.write("\n")
        st.sidebar.write("\n")
        st.sidebar.write("\n")

        #Даны 3 варианта сайта (n_sites)
        #Реальные (то есть не известные аналитику) вероятности покупок составляют:
        #    сайт 1- (на 3% меньше, чем 2-й сайт)
        #    сайт 2 - =1.2%
        #    сайт 3 - (на 2% меньше, чем 2-й сайт)

        #np.random.seed(0) # начальное значение для целей воспроизводимости

        n_1_hour = 10000 # среднее кол-во входов на сайт в течение Одного часа
        SKO_n_1_hour = 0 # СКО среднее кол-во входов на сайт в течение Одного часа

        n_sites = 3 # кол-во сайтов
        n_trials = 300 # кол-во реализаций для набора статистики методом бэггинга 
                # то есть будем повторять работу нашего алгоритма n_trials раз,
                # чтобы можно было рассчитать средние сначения и доверительные интервалы
                # В реальной жизни никаких n_trials реализаций НЕ ДЕЛАЕМ


        #n_hours = 300 # сколько часов тестируем сайты

        P_site_real = np.zeros([n_sites])
        P_site_real[0] = 1.2*0.98/100  #  сайт 1- (на 2% меньше, чем 2-й сайт)
        P_site_real[1] = 1.2/100       #  сайт 2 - =1.2%
        P_site_real[2] = 1.2*0.97/100  #  сайт 3 - (на 3% меньше, чем 2-й сайт)


        Site_opt_real=1 # индекс сайта с max реальной конверсией - Задан в массиве "P_site_real" 

        #Epsilon = 0.1
        array_current= np.zeros([n_sites, 7])
        array_trials = np.zeros([n_trials, n_hours,3])
        array_mean_trials = np.zeros([n_hours,5])
        resuit_show_site = np.zeros([n_sites])



        def display_site (P, N, CKO):

                #    P - вероятность конверсии (покупки) (задано в условии задачи).
                #    N - СРЕДНЕЕ кол-во зашедших за 1 час посетителей на сайт
                #        (задано в условии задачи). 
                #    CKO - Сред. Квадр. Откл. для N (задано в условии задачи).
                
                n_input = int(np.random.normal(N, CKO, 1)) # число входов на сайт в этом часе
                s = np.random.binomial(1, P, n_input) # 1D массив длинной N. Каждый элемент 
                                                        # массива либл 0 либо 1. 1 выпадает с 
                                                        # вероятностью Р 
                n1 = ((s >= 1).sum()) # кол-во единиц в массиве s
                Teta_estimation = n1/n_input


                #    Teta_estimation - оценка P (процент конверций (в долях единицы ) ()
                #    n1 - кол-во конверсий на сайте 
                #    n_input - суммарное кол-во входов на сайт

                return [Teta_estimation, n1, n_input]



        def update_array_curret(i_site_show):

                #    В функцию передаётся индекс сайта, который только что показали.
                #    Функция обновляет массивы "array_current"  и "array_trials".
                #    Эти массивы объявлены в основной программе, след-но это глобальные переменные.
                #    Так же глобальные переменные "i_trial" , "i_hour" и массив "resuit_show_site"

                array_current[i_site_show, 1] += resuit_show_site[1]           
                array_current[i_site_show, 2] += resuit_show_site[2]                              
                #    """в "array_current[i_site_show, 0]" накапливаем процент конверсий за все показы этого сайта"""
                array_current[i_site_show, 0] =  array_current[i_site_show, 1] / array_current[i_site_show, 2]
                t=array_current[:, 2].sum()
                array_current[i_site_show, 3] =  array_current[i_site_show, 0] +  np.sqrt( (2*np.log(t))/array_current[i_site_show, 2])
                array_current[i_site_show, 4] =  array_current[i_site_show, 0] +  np.sqrt( (2*np.log(array_current[i_site_show, 2]))/array_current[i_site_show, 1])
                # В "аrray_current[i_site_show, 5]" дисперсия рассчитывается для оценки Вероятности Успеха (т.е. Единицы) == Корень (P*(1-P)/n) 
                array_current[i_site_show, 5] =  array_current[i_site_show, 0] +  3*np.sqrt( (array_current[i_site_show, 0]*(1-array_current[i_site_show, 0]))/array_current[i_site_show, 2])
                

                #    """ в "array_trials[i_trial, i_hour, 0]" храним текущий процент конверсий  (который  был в "i_hour" часе) """
                array_trials[i_trial, i_hour, 0] =  resuit_show_site[0] 
                array_trials[i_trial, i_hour, 1] =  i_site_show   #какой сайт показывали в этом (текущем) часе. 
                
                array_trials[i_trial, i_hour, 2] =  (array_trials[i_trial, 0:i_hour + 1, 1] == Site_opt_real).sum() / (i_hour + 1) 

                t = i_hour + 1
                n_a=(array_trials[i_trial, 0:i_hour+1, 1] == i_site_show).sum()
                array_current[i_site_show, 6] =  array_current[i_site_show, 0] +  np.sqrt( (2 * np.log(t))/n_a)

                return



        def choice_of_site_for_show():
                # Находим индекс сайта с максимальной оценкой Верхней Оценкой ДИ (UCB)
                i_site_for_show = np.argmax(array_current[:,5], axis=0)
                return (i_site_for_show)



        # цикл по реализациям бэггинга, то есть повторяем работу нашего алгоритма n_trials раз,
        # чтобы можно было рассчитать средние сначения и доверительные интервалы  # В реальной жизни никаких n_trials реализаций НЕ ДЕЛАЕМ
        for i_trial in range(n_trials): 
                # начинаем тестировать сайты.
                i_hour = int(0) # текущтй час равен 0.
                array_current=np.zeros(array_current.shape)

                # показываем каждый сайт по одному разу и таких циклов n_start
                n_start=1 
                if (n_start*n_sites >= n_hours) :
                        print("***---***---******************************************************")
                        print("***---***---******************************************************")
                        print(" ОШИБКА (n_start*n_sites >= n_hours) ==> STOP" )
                        print("***---***---******************************************************")
                        print("***---***---******************************************************")
                        stop
                
                for i in range(n_start):  
                        for i_site in range(n_sites):
                                resuit_show_site = display_site (P_site_real[i_site], n_1_hour, SKO_n_1_hour)
                                # обновляеммм массивы после очередного показа
                                update_array_curret(i_site) 
                                i_hour +=1 # начитнаеися следующий час и переходим к следующему сайту


                array_site = [0, 1, 2] # массив, в который записываются порядок показа сайта
                for i_hour in range(i_hour, n_hours):      
                        # Определяем Сайт для показа 
                        i_site_for_show=choice_of_site_for_show()
                        # Показываем, выбранный сайт
                        resuit_show_site = display_site (P_site_real[i_site_for_show], n_1_hour, SKO_n_1_hour)
                        # Обновляем/Дополнаем массивы с информацией о показах сайтов
                        update_array_curret(i_site_for_show)
                        array_site.append(i_site_for_show)
                
                
                
        array_mean_trials[:,0] = np.mean(array_trials[:,:,0], axis=0)
        array_mean_trials[:,1] = np.mean(array_trials[:,:,2], axis=0)*100
        array_mean_trials[:,2] = np.percentile((array_trials[:,:,2]), 5, axis=0)*100
        array_mean_trials[:,3] = np.percentile((array_trials[:,:,2]), 95, axis=0)*100

        print("--------------------------------------------------------------")
        print("--- UCB1 -----------------------------------------------------")

        print ("array_current (для ПОСЛЕДНЕГО i_trial) =")
        print(array_current)

        print ("array_mean_trials[n_hours-1,1]=",np.int_(array_mean_trials[n_hours-1,1]),"%")
        print ("Доверит. Интервал = ",np.int_(array_mean_trials[n_hours-1,2]), np.int_(array_mean_trials[n_hours-1,3]))
        print("n_hours=",n_hours, " n_trials=",n_trials)

        print ("n_start=",n_start)


        n_trials_plot=2
        Opt=1.2
        fig, ax = plt.subplots()

        x =  np.arange(n_hours)

        ax.set(xlabel='Тактыы', 
        ylabel='Процент конверсии', 
        title='Моделирование работы алгоритма $UCB$')
        colors = ["red",
                "black",
                "magenta",
                "blue",
                "grey",
                ]


        # Рисуем Реализации Разными Цветоми
        for i_trials, color in zip(range(n_trials_plot), colors):
                ax.plot(x,array_trials[i_trials,:,0]*100 , c=color, linestyle='--', linewidth=1)
        
        
        ax.plot(x, array_mean_trials[:,0]*100, c="red", linestyle='solid', lw = 2.5)
        ax.hlines(Opt, xmin=x[0], xmax=x[-1], colors='red', linestyle='--', linewidth=2.5)
        #st.pyplot(fig)


        fig, ax = plt.subplots()

        x =  np.arange(n_hours)

        ax.set(xlabel='Такты', 
        ylabel='Процент конверсии', 
        title='Моделирование работы алгоритма $UCB$')

        ax.set_ylim(0,110) # пределы Оси "Y"

        ax.plot(x, array_mean_trials[:,1], c="red", linestyle='solid', lw = 2.5)
        ax.plot(x, array_mean_trials[:,2], c="red", linestyle='--', lw = 1.5)
        ax.plot(x, array_mean_trials[:,3], c="red", linestyle='--', lw = 1.5)
        ax.hlines(100, xmin=x[0], xmax=x[-1], colors='red', linestyle='--', linewidth=2.5)
        ax.grid(True)
        #st.pyplot(fig)


        fig, ax = plt.subplots()
        ax.set(xlabel='Такты', 
        ylabel='Конверсия', 
        title='Моделирование работы алгоритма Верхнего доверительного интервала')
        colors = ["red",
                "black",
                "magenta",
                "blue",
                "grey",
                ]

        x =  np.arange(n_hours)
        y = []


        blue_patch = mpatches.Patch(color = 'blue', label = 'Баннер 0')
        green_patch = mpatches.Patch(color = 'green', label = 'Баннер 1')
        magenta_patch = mpatches.Patch(color = 'magenta', label = 'Баннер 2')
        ax.legend(handles = [blue_patch, green_patch, magenta_patch])
        ax.plot(x, array_mean_trials[:,0]*10, c = "grey", linestyle = 'solid', lw = 1.5)
        ax.grid(True)

        i = 0
        while i < n_hours:
                # точки для графика
                if int(array_site[i]) == 0:
                        ax.plot(i, array_mean_trials[:,0][i]*10, "bo")
                if int(array_site[i]) == 1:
                        ax.plot(i, array_mean_trials[:,0][i]*10, "go")
                if int(array_site[i]) == 2:
                        ax.plot(i, array_mean_trials[:,0][i]*10, "mo")
                
                i += 1

        st.pyplot(fig)
        #st.button("Re-run")
        
else:

        st.write("# Алгоритм верхнего доверительного интервала")

        st.markdown(
        """
Алгоритм верхнего доверительного интервала (Upper Confidence Bound) является одним из популярных 
алгоритмов многоруких бандитов, который используется для принятия решений в проблеме исследования и использования. 

Основная идея алгоритма заключается в том, чтобы выбирать действие с наивысшей верхней границей доверительного интервала. 
Доверительный интервал для каждого действия рассчитывается на основе уже полученных наблюдений и количества раз, 
которое это действие было выбрано в прошлом.

Алгоритм функционирует следующим образом:
1. Инициализация: каждое действие считается равновероятным и выбирается по одному разу
2. Обновление среднего значения и доверительного интервала: после выполнения каждого действия рассчитывается его среднее значение вознаграждения и обновляется доверительный интервал
3. Выбор действия: выбирается действие с наивысшей верхней границей доверительного интервала, которое позволяет более эффективно исследовать и использовать наиболее перспективные действия

""")
        

        from PIL import Image
        image = Image.open('5_1.png')
        st.image(image)
        
        st.write("## Пример ")
        
        image = Image.open('5_2.png')
        st.image(image)
        st.markdown(
                """
Находим оптимальный баннер (с максимальным UCB) – это Баннер-2

Показали 10000 пользователям

Кликов было 145

Пересчитываем строку 2 и обновляем таблицу:

                """
        )
        image = Image.open('5_3.png')
        st.image(image)
        st.markdown(
                """
Для следующего показа выбираем баннер с максимальным UCB – Баннер 3

                """
        )