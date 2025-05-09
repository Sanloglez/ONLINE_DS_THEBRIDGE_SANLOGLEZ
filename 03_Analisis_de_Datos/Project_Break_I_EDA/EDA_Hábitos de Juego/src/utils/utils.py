import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Función para graficar columnas categóricas con el mismo color
BASE_COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#b3b3cc', '#f781bf']
def graficar_columna_categorica(df, columna, titulo=None):
    """
    Genera un gráfico de barras con colores diferenciados y sin leyenda,
    adaptando los colores al número de categorías de la columna.

    Parámetros:
        df: DataFrame de entrada
        columna: str, nombre de la columna categórica a graficar
        titulo: str, título del gráfico
    """
    # Contar valores de la columna
    conteo = df[columna].value_counts()
    categorias_colors = conteo.index.tolist()

    # Ajustar la paleta según número de categorías
    colores = BASE_COLORS[:len(categorias_colors)]

    # Crear gráfico
    plt.figure(figsize=(8, 4))
    plt.bar(categorias_colors, conteo.values, color=colores)
    
    # Estética
    plt.title(titulo or f'{columna} distribution')
    plt.xlabel(columna.replace('_', ' ').title())
    plt.ylabel('Players')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Función para unificar los valores de "devices" y limpiarlos
def limpiar_dispositivos_orden_personalizado(col):
    orden = ['PC', 'Console', 'Mobile']
    mapa_final = {'pc': 'PC', 'console': 'Console', 'mobile': 'Mobile'}

    def procesar(x):
        if pd.isna(x):
            return None
        x = str(x).lower()
        x = x.replace('console (playstation, xbox, etc.)', 'console')
        x = x.replace('handheld devices (nintendo switch, etc.)', 'console')
        x = x.replace('tablet', 'mobile')
        x = x.replace('all', 'pc, mobile, console')
        dispositivos_raw = [d.strip() for d in x.replace(', ', ',').replace(' ,', ',').split(',')]
        dispositivos = [mapa_final.get(d, d.capitalize()) for d in dispositivos_raw]
        dispositivos = list(set(dispositivos))  # eliminar duplicados
        dispositivos_ordenados = [d for d in orden if d in dispositivos]
        return ', '.join(dispositivos_ordenados)
    

# Función para recombrar columnas repetidas con sufijos
def renombrar_repetidas_con_sufijos(df, mantener_sin_sufijo=None):
    if mantener_sin_sufijo is None:
        mantener_sin_sufijo = []

    nombre_counts = {}
    nuevos_nombres = []

    for col in df.columns:
        if col in mantener_sin_sufijo:
            nuevos_nombres.append(col)
        else:
            if col not in nombre_counts:
                nombre_counts[col] = 1
                nuevos_nombres.append(f"{col}_1")
            else:
                nombre_counts[col] += 1
                nuevos_nombres.append(f"{col}_{nombre_counts[col]}")

    df.columns = nuevos_nombres
    return df

# Función de limpieza de géneros
mapa_generos = {
    'first-person shooter (fps)': 'First-Person Shooter',
    'fps': 'First-Person Shooter',
    'action/adventure': 'Action-Adventure',
    'action adventure': 'Action-Adventure',
    'puzzle/strategy': 'Puzzle-Strategy',
    'simulation (e.g., the sims)': 'Simulation',
    'simulation': 'Simulation',
    'sports': 'Sports',
    'mmo (massively multiplayer online)': 'MMO',
    'role-playing games (rpg)': 'Role-Playing',
    'role-playing': 'Role-Playing',
    'rpg': 'Role-Playing',
    'casual': 'Casual',
    'horror': 'Horror',
    # en caso de que alguien haya escrito todo junto
    'action/adventure, sports': 'Action-Adventure, Sports'
}

def limpiar_generos(col):
    def procesar(x):
        if pd.isna(x):
            return None
        x = str(x).lower()
        generos_raw = [g.strip() for g in x.replace(', ', ',').replace(' ,', ',').split(',')]
        generos_norm = [mapa_generos.get(g, g.title()) for g in generos_raw]
        generos_unicos = sorted(set(generos_norm))
        return ', '.join(generos_unicos)
    
    return col.apply(procesar)

# Función de limpieza de juegos favoritos
mapa_juegos = {
    'Call Of Duty': 'Call Of Duty',
    'CALL OF DUTY': 'Call Of Duty',
    'call of duty': 'Call Of Duty',

    'Fornite': 'Fortnite',
    'Fortnite': 'Fortnite',

    'efootball': 'Efootball',
    'Efootball': 'Efootball',

    'FC MOBILE': 'FC Mobile',
    'Fc Mobile': 'FC Mobile',

    'Solo leveling arise': 'Solo Leveling',
    'Solo Levelling': 'Solo Leveling',

    'Wuthering waves': 'Wuthering Waves',
    'Wuther Waves': 'Wuthering Waves',

    'Rhythm Rush lite': 'Rhythm Rush Lite',
    'Wukong': 'Wukong',
    'Subway': 'Subway',
    'Many': 'Many',
    'BGMI': 'BGMI',
    'bgmi,coc,chess': 'BGMI, COC, Chess',
    'Moba Legends': 'MOBA Legends',
    'Red dead redemption 2': 'Red Dead Redemption 2',
    'God of war ragnarok': 'God Of War Ragnarok',
    'Chess and clash of clans': 'Chess And Clash Of Clans',
    'Free fire,wuthering waves': 'Free Fire, Wuthering Waves',

    # Juegos del favorite_game_2
    'FIFA 2024': 'FIFA',
    'Call of Duty': 'Call Of Duty',
    'Overwatch': 'Overwatch',
    'League of Legends': 'League Of Legends',
    'Minecraft': 'Minecraft',
    'Genshin Impact': 'Genshin Impact',
    'Apex Legends': 'Apex Legends',
    'Among Us': 'Among Us',
    'Valorant': 'Valorant'
}

# Función para limpiar y mapear juegos favoritos
def limpiar_favorite_game(col):
    return col.str.strip().str.title().replace(mapa_juegos)

# Función para limpiar y mapear motivaciones
mapa_motivation = {
    'for fun/entertainment': 'Fun',
    'to relieve stress': 'Stress Relief',
    'to socialize': 'Socializing',
    'to socialize with friends': 'Socializing',
    'to improve skills/competition': 'Competition',
    "learning how it's designed": 'Learning',
    'for the story/experience': 'Story',
    'if no other better work': 'Other'
}

def limpiar_motivation(col):
    def procesar(x):
        if pd.isna(x):
            return None
        x = str(x).lower()
        items = [i.strip() for i in x.replace(', ', ',').replace(' ,', ',').split(',')]
        normalizados = [mapa_motivation.get(i, i.title()) for i in items]
        return ', '.join(sorted(set(normalizados)))
    return col.apply(procesar)

# Función definir cardinalidad
def card_tipo(df,umbral_categoria = 10, umbral_continua = 30):
    df_temp = pd.DataFrame([df.nunique(), df.nunique()/len(df) * 100, df.dtypes]) 
    df_temp = df_temp.T 
    df_temp = df_temp.rename(columns = {0: "Card", 1: "%_Card", 2: "Tipo"}) 

    df_temp.loc[df_temp.Card == 1, "%_Card"] = 0.00

   
    df_temp["tipo_sugerido"] = "Categorica"
    df_temp.loc[df_temp["Card"] == 2, "tipo_sugerido"] = "Binaria"
    df_temp.loc[df_temp["Card"] >= umbral_categoria, "tipo_sugerido"] = "Numerica discreta"
    df_temp.loc[df_temp["%_Card"] >= umbral_continua, "tipo_sugerido"] = "Numerica continua"


    return df_temp