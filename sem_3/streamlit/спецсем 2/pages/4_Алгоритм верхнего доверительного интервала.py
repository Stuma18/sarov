import streamlit as st

st.set_page_config(page_title = "Алгоритм верхнего доверительного интервала")

st.sidebar.success("Алгоритм верхнего доверительного интервала")

st.write("# Алгоритм верхнего доверительного интервала")

st.markdown(
        """
Алгоритм верхнего доверительного интервала (Upper Confidence Bound (UCB)) является одним из популярных 
алгоритмов многоруких бандитов, который используется для принятия решений в проблеме исследования и использования. 
Он может быть применен в различных областях, включая рекомендательные системы, интернет-маркетинг, клинические испытания и другие.

Основная идея UCB заключается в том, чтобы выбирать действие с наивысшей верхней границей доверительного интервала. 
Доверительный интервал для каждого действия рассчитывается на основе уже полученных наблюдений и количества раз, 
которое это действие было выбрано в прошлом.

Алгоритм UCB функционирует следующим образом:
1. Инициализация: каждое действие считается равновероятным и выбирается по одному разу
2. Обновление среднего значения и доверительного интервала: после выполнения каждого действия рассчитывается его среднее значение вознаграждения и обновляется доверительный интервал
3. Выбор действия: выбирается действие с наивысшей верхней границей доверительного интервала, которое позволяет более эффективно исследовать и использовать наиболее перспективные действия

Алгоритм UCB имеет преимущество в балансировке исследования и использования. Он стремится выбирать действия, 
которые имеют самую высокую вероятность оптимальности исходя из имеющихся данных. При этом он также продолжает 
исследовать другие действия для получения дополнительной информации и улучшения решений в долгосрочной перспективе.

""")

from PIL import Image
image = Image.open('4.png')
st.image(image)