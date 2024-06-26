import streamlit as st

st.set_page_config(page_title = "Multi-armed bandit")

st.sidebar.success("Multi-armed bandit")

st.write("# Multi-armed bandit")

st.markdown(
        """
Многорукие бандиты — это класс алгоритмов машинного обучения, которые используются для 
решения задач оптимизации при принятии решений в условиях неопределенности, другими словами, 
в ситуациях, где имеется несколько вариантов выбора и каждый выбор имеет свою вероятность успеха. 

В целом, алгоритмы многорукого бандита отличаются от других алгоритмов тем, что предлагают 
эффективный и адаптивный способ управления риском и повышения эффективности бизнес-процессов.

Существует несколько различных алгоритмов многоруких бандитов, которые 
могут быть использованы для разных видов задач. Например:
1. E-greedy алгоритм;
2. Upper Confidence Bound (UCB);
3. Thompson Sampling (TS).

""")