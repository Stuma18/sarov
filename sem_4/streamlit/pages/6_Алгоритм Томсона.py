import streamlit as st

st.set_page_config(page_title = "Алгоритм Томсона")

st.sidebar.success("Алгоритм Томсона")

#st.sidebar.button("Описание алгоритма")
#if st.sidebar.button('Моделирование работы алгоритма'):
        
status = st.sidebar.radio("Выберете: ", ("Описание алгоритма Томсона", 'Моделирование работы алгоритма Томсона'))
if (status == 'Моделирование работы алгоритма Томсона'):
        
        st.write("# Моделирование работы алгоритма Томсона")

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        #Даны 3 варианта сайта (n_sites)
        #Реальные (то есть не известные аналитику) вероятности покупок составляют:
        #сайт 1- (на 2% меньше, чем 2-й сайт)
        #сайт 2 - =1.2%
        #сайт 3 - (на 1% меньше, чем 2-й сайт)


        #np.random.seed(0) # начальное значение для целей воспроизводимости

        n_1_hour = 10000 # среднее кол-во входов на сайт в течение Одного часа
        SKO_n_1_hour = 1000 # СКО среднее кол-во входов на сайт в течение Одного часа

        n_sites = 3 # кол-во сайтов
        n_trials = 300
        # кол-во реализаций для набора статистики методом бэггинга 

        n_hours = st.sidebar.slider('Количество тактов:', 10, 400, 100, step = 10)
        #n_hours = 2000 # сколько часов тестируем сайты
        
        st.sidebar.write("\n")
        st.sidebar.write("\n")
        st.sidebar.write("\n")
        st.sidebar.write("\n")

        P_site_real = np.zeros([n_sites])
        P_site_real[0] = 1.2*0.98/100  #  сайт 1- (на 2% меньше, чем 2-й сайт)
        P_site_real[1] = 1.2/100       #  сайт 2 - =1.2%
        P_site_real[2] = 1.2*0.97/100  #  сайт 3 - (на 3% меньше, чем 2-й сайт)

        Site_opt_real=1 # индекс сайта с max реальной конверсией - Задан в массиве "P_site_real" 

        array_current= np.zeros([n_sites, 6])
        array_trials = np.zeros([n_trials, n_hours,6])
        array_mean_trials = np.zeros([n_hours,5])
        resuit_show_site = np.zeros([n_sites])


        #print("** Start array_current=")
        #print(array_current)
        #stop

        def display_site (P, N, CKO):
                     
                n_input = int(np.random.normal(N, CKO, 1)) # число входов на сайт в этом часе
                s = np.random.binomial(1, P, n_input) # 1D массив длинной N. Каждый элемент 
                #print(s)                        # массива либл 0 либо 1. 1 выпадает с 
                                                # вероятностью Р 
                n1 = ((s >= 1).sum()) # кол-во единиц в массиве s
                Teta_estimation = n1/n_input # ;   print("From display_site  P_real=",P,",  ",Teta_estimation,",  ",n1,",  ",n_input)

                
                return [Teta_estimation, n1, n_input]


        def update_array_curret(i_site_show):
               

                array_current[i_site_show, 1] += resuit_show_site[1] # = n1 - сколько было конверсий за время i_hour  
                array_current[i_site_show, 2] += resuit_show_site[2] # = N - сколько пользователей видели сайт (за время i_hour)                     
                array_current[i_site_show, 0] =  array_current[i_site_show, 1] / array_current[i_site_show, 2]
                
                array_current[i_site_show, 3] =  array_current[i_site_show, 1] # = Alf, т.е. обновляем Alf, показанного сайта
                array_current[i_site_show, 4] =  array_current[i_site_show, 2]-array_current[i_site_show, 1] # = Bet, т.е. обновляем Bet, показанного сайта
                #Сэмплируем показанный сайт
                #array_current[i_site_show, 5] = np.random.beta(array_current[i_site_show, 3], array_current[i_site_show, 4])
                array_current[:, 5] = np.random.beta(array_current[:, 3], array_current[:, 4],3) # в 5-й столбец заносим новые сэмпля для кажного Баннера
                
               
                array_trials[i_trial, i_hour, 0] =  resuit_show_site[0] 
                array_trials[i_trial, i_hour, 1] =  i_site_show   #какой сайт показывали в этом (текущем) часе. 
                
                array_trials[i_trial, i_hour, 2] =  (array_trials[i_trial, 0:i_hour+1, 1] == Site_opt_real).sum() / (i_hour+1) 

                array_trials[i_trial, i_hour, 3:7] = array_current[:, 0] 


                return    
        
        def choice_of_site_for_show():  
                # Находим индекс сайта с максимальной сэмплированной величиной P_конверсии
                i_site_for_show = np.argmax(array_current[:,5], axis=0)
                #print(" ******** From choice_of_site_for_show")
                #print("i_site_for_show=",i_site_for_show)
                return (i_site_for_show)
        

        # цикл по реализациям бэггинга
        for i_trial in range(n_trials): 
                #print("*** for i_trial in range(n_trials): i_trial =", i_trial  )
                # начинаем тестировать сайты.
                i_hour = int(0) # текущтй час равен 0.
                
                array_current=np.zeros(array_current.shape) #;print ("0** array_current="); print(array_current)
                array_current[:,3:5] = np.ones((3,2))  #заполняем Alf=1 и Bet=1 
                
                #Первичное Сэмплирование Каждого Сайта
                #print("Перед=")
                #print(array_current)
                array_current[:, 5] = np.random.beta(array_current[:, 3], array_current[:, 4],3)
                #print("После=")
                #print(array_current)
                #stop


                n_start=1
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
                        
                #print("i_trial=",i_trial)
                #print ("array_current=")
                #print(array_current)

        array_mean_trials[:,0] = np.mean(array_trials[:,:,0], axis=0)
        array_mean_trials[:,1] = np.mean(array_trials[:,:,2], axis=0)*100
        array_mean_trials[:,2] = np.percentile((array_trials[:,:,2]), 5, axis=0)*100
        array_mean_trials[:,3] = np.percentile((array_trials[:,:,2]), 95, axis=0)*100

        print("--------------------------------------------------------------")
        print("--- Thompson Sampling ----------------------------------------------")

        print ("array_current (для ПОСЛЕДНЕГО i_trial) =")
        print(array_current)

        #print ("array_mean_trials[n_hours-1,1]=",np.int(array_mean_trials[n_hours-1,1]),"%")
        #print ("Доверит. Интервал = ",np.int(array_mean_trials[n_hours-1,2]), np.int(array_mean_trials[n_hours-1,3]))
        #print("n_hours=",n_hours, " n_trials=",n_trials)



        n_trials_plot=2
        Opt=1.2
        #for i_trials, color in zip(range(n_trials), colors): 
        #ax.plot(x, array_trials2[i_trials,:], c=color, linestyle='--', linewidth=1)
        fig, ax = plt.subplots()

        x =  np.arange(n_hours)

        ax.set(xlabel='Часы (hours)', 
                ylabel='% конверсий (CTR=???)', 
                title='Средний по n_trials %Конверсий и Первые n конверсий $(Thompson-Sampling$)')
        colors = ["red",
                "black",
                "magenta",
                "blue",
                "grey",
                ]

        #ax.set_ylim(1.15,1.25) # пределы Оси "Y"

        # Рисуем Реализации Разными Цветоми
        for i_trials, color in zip(range(n_trials_plot), colors):
                ax.plot(x,array_trials[i_trials,:,0]*100 , c=color, linestyle='--', linewidth=1)
        
        # Рисуем Реализации Одним Цветом
        #for i_trials in range(n_trials_plot):    
        #    ax.plot(x,array_trials[i_trials,:,0]*100 , c="grey", linestyle='--', linewidth=1)
        
        
        ax.plot(x, array_mean_trials[:,0]*100, c="red", linestyle='solid', lw = 2.5)
        ax.hlines(Opt, xmin=x[0], xmax=x[-1], colors='red', linestyle='--', linewidth=2.5)
        plt.show()


        fig, ax = plt.subplots()

        x =  np.arange(n_hours)

        ax.set(xlabel='Часы (hours)', 
                ylabel='% конверсий (CTR=???)', 
                title='Средний % Показа Opt-сайта и ДИ $(Thompson-Sampling)$')
        ax.set_ylim(0,110) # пределы Оси "Y"


        ax.plot(x, array_mean_trials[:,1], c="red", linestyle='solid', lw = 2.5)

        ax.plot(x, array_mean_trials[:,2], c="red", linestyle='--', lw = 1.5)
        ax.plot(x, array_mean_trials[:,3], c="red", linestyle='--', lw = 1.5)

        ax.hlines(100, xmin=x[0], xmax=x[-1], colors='red', linestyle='--', linewidth=2.5)
        ax.grid(True)
        plt.show()
        
        
        
        
        
        fig, ax = plt.subplots()
        ax.set(xlabel='Такты', 
        ylabel='Конверсия', 
        title='Моделирование работы алгоритма Томсона')
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

        st.write("# Алгоритм Томсона")

        st.markdown(
        """Бета распределение""")
        st.latex(r'''
    f(x)=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}
    ''')
        st.markdown(
        """где""")
        st.latex(r'''
    \alpha,\beta\gt 0 - \text{параметры}   
    ''')
        st.latex(r'''
    B(\alpha,\beta)=\int_{0}^{1}x^{\alpha-1}(1-x)^{\beta-1}dx - \text{бета функция} 
    ''')
        st.latex(r'''
    x\in[0,1]
    ''')

        from PIL import Image
        image = Image.open('6_1.png')
        st.image(image)
        
        image = Image.open('6_2.png')
        st.image(image)
        
        st.markdown(""" Шаг 1:""")
        image = Image.open('6_3.png')
        st.image(image)
        
        st.write(r""" Эти нули – это наши текущие значения параметров бета-распределения (то есть значения $\alpha$ и $\beta$)""")
                    
        st.write(r"""При $\alpha = 0$ и $\beta = 0$ бета-распределение представляет собой равномерное в интервале $[0,1]$""")
        
        #image = Image.open('6_4.png')
        #st.image(image)
        st.write(r"""Генерируем 3 случайных числа из бета-распределения с текущими параметрами $\alpha = 0$ и $\beta = 0$ и заносим их в самый правый столбец: """)

        image = Image.open('6_5.png')
        st.image(image)
        
        st.write(r"""Выбираем для показа баннер с максимальным Сэмплом  - это Банер 1""")
        st.write(r"""После показа Баннера 1 обновляем его строку в таблице (cэмплы остались от предыдущего шага, их еще не пересчитывали):""")

        image = Image.open('6_6.png')
        st.image(image)
        st.write(r"""Следующим шагом будем сэмплировать каждое бета-распределение:""")

        image = Image.open('6_7.png')
        st.image(image)
        st.write(r"""Баннер 1 –  минимальный по сравнению с равномерными распределениями

Баннер 2 и 3 были равномерные распределения – получили довольно большие значения

Максимальный сэмпл у Баннера-2 $\Rightarrow$ его показываем
""")

        
        
