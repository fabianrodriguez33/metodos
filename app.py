import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sympy import symbols, diff, sympify, lambdify

# Configuración de la página
st.set_page_config(
    page_title="Métodos Numéricos - Raíces de Ecuaciones",
    page_icon="🧮",
    layout="wide"
)

# Título principal
st.title("🧮 Raíces de Ecuaciones No Lineales")
st.markdown("**Sesión 3: Método del Punto Fijo, Newton-Raphson y Secante**")

# Barra lateral para configuración
st.sidebar.header("⚙️ Configuración")

# Selección de método
metodo = st.sidebar.selectbox(
    "Selecciona el método:",
    ["Punto Fijo", "Newton-Raphson", "Secante"],
    index=0
)

# Selección de problema predefinido o personalizado
opcion = st.sidebar.radio(
    "Usar problema:",
    ["Problemas de la guía", "Función personalizada"]
)

# Funciones predefinidas según el método
if opcion == "Problemas de la guía":
    if metodo == "Punto Fijo":
        st.sidebar.info("Problema 1: Control de temperatura")
        funcion_str = "18 + 8*exp(-0.15*x)"
        x0_default = 20.0
        x1_default = None
    elif metodo == "Newton-Raphson":
        st.sidebar.info("Problema 2: Sistema de almacenamiento")
        funcion_str = "x**3 - 7*x - 5"
        x0_default = 3.0
        x1_default = None
    else:  # Secante
        st.sidebar.info("Problema 3: Rendimiento de servidor")
        funcion_str = "exp(-x) - x**2 + 0.2"
        x0_default = 0.0
        x1_default = 1.0
else:
    funcion_str = st.sidebar.text_input(
        "Ingresa la función f(x):",
        value="x**3 - 7*x - 5",
        help="Usa sintaxis de Python. Ejemplo: x**2 - 4, exp(-x) - x, sin(x) - x/2"
    )
    x0_default = 1.0
    x1_default = 2.0

# Parámetros de entrada
st.sidebar.subheader("📊 Parámetros")

if metodo == "Secante":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x0 = st.number_input("x₀ (inicial):", value=x0_default, format="%.4f")
    with col2:
        x1 = st.number_input("x₁ (segundo):", value=x1_default, format="%.4f")
else:
    x0 = st.number_input("Valor inicial x₀:", value=x0_default, format="%.4f")
    x1 = None

# Tolerancia y máximo de iteraciones
epsilon = st.sidebar.slider("Tolerancia εa (%):", 0.001, 1.0, 0.01, 0.001) / 100
max_iter = st.sidebar.slider("Máximo de iteraciones:", 10, 100, 50, 5)

# Función para evaluar la expresión
def evaluar_funcion(func_str, x_val):
    try:
        x = symbols('x')
        expr = sympify(func_str)
        f = lambdify(x, expr, modules=['numpy', 'math'])
        return f(x_val)
    except Exception as e:
        st.error(f"Error en la función: {e}")
        return None

# Función para calcular derivada simbólica
def calcular_derivada(func_str):
    try:
        x = symbols('x')
        expr = sympify(func_str)
        derivada = diff(expr, x)
        return str(derivada)
    except:
        return None

# MÉTODO DEL PUNTO FIJO
def punto_fijo(func_str, x0, epsilon, max_iter):
    """
    Método del Punto Fijo: x_{n+1} = g(x_n)
    """
    x = symbols('x')
    g_expr = sympify(func_str)
    g = lambdify(x, g_expr, modules=['numpy', 'math'])
    
    # Calcular derivada de g para verificar convergencia
    g_prime_expr = diff(g_expr, x)
    g_prime = lambdify(x, g_prime_expr, modules=['numpy', 'math'])
    
    iteraciones = []
    x_n = x0
    
    for i in range(max_iter):
        try:
            x_n1 = g(x_n)
            f_x = x_n - x_n1  # f(x) = x - g(x)
            
            if i == 0:
                error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0
            else:
                error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0
            
            iteraciones.append({
                'Iteración': i,
                'x_n': x_n,
                'f(x_n)': f_x,
                'Error %': error * 100
            })
            
            if error <= epsilon:
                break
            
            x_n = x_n1
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break
    
    return pd.DataFrame(iteraciones), x_n1, g_prime(x_n1)

# MÉTODO DE NEWTON-RAPHSON
def newton_raphson(func_str, x0, epsilon, max_iter):
    """
    Método de Newton-Raphson: x_{n+1} = x_n - f(x_n)/f'(x_n)
    """
    x = symbols('x')
    f_expr = sympify(func_str)
    f = lambdify(x, f_expr, modules=['numpy', 'math'])
    
    f_prime_expr = diff(f_expr, x)
    f_prime = lambdify(x, f_prime_expr, modules=['numpy', 'math'])
    
    iteraciones = []
    x_n = x0
    
    for i in range(max_iter):
        try:
            f_x = f(x_n)
            f_prime_x = f_prime(x_n)
            
            if abs(f_prime_x) < 1e-10:
                st.warning(f"⚠️ Derivada muy pequeña en iteración {i}")
                break
            
            x_n1 = x_n - f_x / f_prime_x
            
            if i == 0:
                error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0
            else:
                error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0
            
            iteraciones.append({
                'Iteración': i,
                'x_n': x_n,
                'f(x_n)': f_x,
                'Error %': error * 100
            })
            
            if error <= epsilon:
                break
            
            x_n = x_n1
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break
    
    return pd.DataFrame(iteraciones), x_n1, str(f_prime_expr)

# MÉTODO DE LA SECANTE
def secante(func_str, x0, x1, epsilon, max_iter):
    """
    Método de la Secante: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    """
    x = symbols('x')
    f_expr = sympify(func_str)
    f = lambdify(x, f_expr, modules=['numpy', 'math'])
    
    iteraciones = []
    x_n_1 = x0
    x_n = x1
    
    for i in range(max_iter):
        try:
            f_x_n_1 = f(x_n_1)
            f_x_n = f(x_n)
            
            denominador = f_x_n - f_x_n_1
            
            if abs(denominador) < 1e-10:
                st.warning(f"⚠️ Denominador muy pequeño en iteración {i}")
                break
            
            x_n1 = x_n - f_x_n * (x_n - x_n_1) / denominador
            
            if i == 0:
                error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0
            else:
                error = abs(x_n1 - x_n) / abs(x_n1) if x_n1 != 0 else 0
            
            iteraciones.append({
                'Iteración': i,
                'x_n': x_n,
                'f(x_n)': f_x_n,
                'Error %': error * 100
            })
            
            if error <= epsilon:
                break
            
            x_n_1 = x_n
            x_n = x_n1
        except Exception as e:
            st.error(f"Error en iteración {i}: {e}")
            break
    
    return pd.DataFrame(iteraciones), x_n1

# Ejecutar el método seleccionado
st.markdown("---")

if st.sidebar.button("🚀 Calcular", type="primary", use_container_width=True):
    
    st.subheader(f"📈 Resultados - Método: {metodo}")
    
    # Ejecutar método
    if metodo == "Punto Fijo":
        df, raiz, g_prime_val = punto_fijo(funcion_str, x0, epsilon, max_iter)
        
        st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
        st.info(f"📐 **Función de iteración:** g(x) = {funcion_str}")
        st.info(f"📐 **Derivada g'(x) en la raíz:** {abs(g_prime_val):.6f}")
        
        if abs(g_prime_val) < 1:
            st.success("✅ **Convergencia garantizada:** |g'(x)| < 1")
        else:
            st.error("❌ **No converge:** |g'(x)| ≥ 1")
    
    elif metodo == "Newton-Raphson":
        df, raiz, derivada = newton_raphson(funcion_str, x0, epsilon, max_iter)
        
        st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
        st.info(f"📐 **Función:** f(x) = {funcion_str}")
        st.info(f"📐 **Derivada:** f'(x) = {derivada}")
    
    else:  # Secante
        df, raiz = secante(func_str, x0, x1, epsilon, max_iter)
        
        st.success(f"✅ **Raíz encontrada:** {raiz:.6f}")
        st.info(f"📐 **Función:** f(x) = {funcion_str}")
    
    # Mostrar tabla de iteraciones
    st.markdown("### 📋 Tabla de Iteraciones")
    st.dataframe(df, use_container_width=True)
    
    # Gráfica de convergencia
    st.markdown("### 📊 Gráfica de Convergencia")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df['Iteración'], df['Error %'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteración', fontsize=12)
        ax1.set_ylabel('Error Relativo (%)', fontsize=12)
        ax1.set_title('Error vs Iteración', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=epsilon*100, color='r', linestyle='--', label=f'Tolerancia ({epsilon*100:.3f}%)')
        ax1.legend()
        st.pyplot(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(df['Iteración'], df['x_n'], 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Iteración', fontsize=12)
        ax2.set_ylabel('Valor de x', fontsize=12)
        ax2.set_title('Convergencia de x', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
    
    # Análisis de resultados
    st.markdown("### 🔍 Análisis de Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Iteraciones", len(df))
    
    with col2:
        st.metric("Error Final", f"{df['Error %'].iloc[-1]:.6f}%")
    
    with col3:
        st.metric("Raíz Aproximada", f"{raiz:.6f}")
    
    # Interpretación según el problema
    if opcion == "Problemas de la guía":
        st.markdown("#### 📝 Interpretación Física")
        if metodo == "Punto Fijo":
            st.info(f"La temperatura de equilibrio del centro de datos es **{raiz:.2f}°C**, dentro del rango óptimo para servidores (18-27°C).")
        elif metodo == "Newton-Raphson":
            st.info(f"El tiempo de respuesta del sistema de almacenamiento es **{raiz:.2f} ms**, indicando una condición operativa eficiente.")
        else:
            st.info(f"El nivel de carga normalizado del servidor es **{raiz:.2f}**, representando el punto óptimo de operación.")

# Información adicional
st.markdown("---")
st.markdown("""
### ℹ️ Información de los Métodos

| Método | Orden de Convergencia | Requiere Derivada | Velocidad |
|--------|----------------------|-------------------|-----------|
| Punto Fijo | Lineal (1) | No | Lenta |
| Newton-Raphson | Cuadrática (2) | Sí | Muy rápida |
| Secante | Superlineal (1.618) | No | Rápida |
""")

# Footer
st.markdown("---")
st.markdown("**Curso:** Métodos Numéricos | **Sesión 3** | Docente: Jorge Luis Manrique Plasencia")