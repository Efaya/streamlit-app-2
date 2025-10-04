import streamlit as st
import pandas as pd

st.title("Мини-программа из Colab")

a = st.number_input("Введите первое число", step=1)
b = st.number_input("Введите второе число", step=1)

if st.button("Выполнить"):
    result = a + b  # Здесь можно заменить на свой код

    df = pd.DataFrame({
        "a": [a],
        "b": [b],
        "result": [result]
    })

    st.subheader("Результат:")
    st.write(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name="result.csv",
        mime="text/csv"
    )
