
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import streamlit as st
#from dbscan_lib import dbscanAlgo as dbs


def dash():
    """ image = Image.open('../report/clusters.png')
    st.image(image, caption = 'Clusters')
     """
     
    tab1, tab2 = st.tabs(["Calibração", "Sessão"])
    
    # To cluster
    with tab1:
        st.header("Calibração")
        with st.expander("Entendea o gráfico"):
                st.write("Estes são os principais pontos onde o usuário manteve o seu olhar fixo durante a calibração do sistema.")
    
        with st.container():
            st.image("../report/clusters.png")
            with st.expander("Entendea o gráfico"):
                st.write("Estes são os principais pontos onde o usuário manteve o seu olhar fixo.")
    
    # To gaze points
    with tab2:
        st.header("Gaze Points")
    
        with st.container():
            st.image("../report/gaze_points.png")   
            with st.expander("Entendea o gráfico"):
                st.write("Está é a trajetória que o usuário fez ao olhar para a imagem.")
                
    # To heatmaps        
    with tab3:
        st.header("Heatmap")
    
        with st.container():
            st.image("../report/heatmap.png")  
            with st.expander("Entendea o gráfico"):
                st.write("A coloração mais intensa, representa o ponto de maior fixação que o usário manteve o seu foco.") 
    
    # To Overlays Images
    with tab4:
        st.header("Overlay Image")
    
        with st.container():
            st.image("../report/overlay_image.png")   
            with st.expander("Entendea o gráfico"):
                st.write("Esta é a representação gráfica da sobreposição entre o mapa de calor e a imagem que o usuário viu.")
         
    
if __name__ == '__main__':
    dash()