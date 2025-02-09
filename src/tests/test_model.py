import pandas as pd
import pickle
import json
import sys
import os

# Adicione o caminho do diretório pai ao sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.utils import import_data
from src.model.MultiArmedBandit import MultiArmedBandit

# Use um caminho absoluto para garantir que o Python encontre os arquivos corretamente
USERS_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'treino')  # Caminho para dados
NEWS_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'itens')
MAPPING_INDEX_PATH = os.path.join(PROJECT_ROOT, 'src', 'utils', 'mapping_index.json')
MAB_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained_models', 'mab.pkl')


def join_data(df_users, df_news):
    # Explodir a coluna 'history' para ter uma linha por news_id
    df_users_exploded = df_users.assign(history=df_users['history'].str.split(', ')).explode('history')

    # Fazer o merge com df_news para obter os títulos
    df_joined = pd.merge(
        df_users_exploded,
        df_news,
        left_on='history',
        right_on='page',
        how='left'
    )
    return df_joined


def load_pkl(mapping_index_path, mab_path):
    # Carregar o modelo MAB
    with open(mab_path, 'rb') as f:
        mab = pickle.load(f)
    # Carregar o mapeamento de news_id para índices
    with open(mapping_index_path, 'r') as f:
        mapping_index = json.load(f)
    return mapping_index, mab


def recommend_news(user_history,
                   mab,
                   mapping_index,
                   news_id_to_title,  # Novo parâmetro
                   top_n=5):
    """
    Recomenda as top-n notícias (títulos) para um usuário, excluindo as já vistas.
    """
    # Converter news_id para índices usando o mapeamento
    user_indices = [mapping_index.get(news_id, -1) for news_id in user_history]

    # Selecionar braços disponíveis (não vistos)
    available_arms = [
        idx for idx in mapping_index.values()
        if idx not in user_indices and idx != -1
    ]

    # Ordenar por UCB e selecionar top-n
    recommended_arms = sorted(
        available_arms,
        key=lambda x: mab.values[x],
        reverse=True
    )[:top_n]

    # Converter índices de volta para news_id
    inverse_mapping = {v: k for k, v in mapping_index.items()}
    recommended_news_ids = [inverse_mapping[idx] for idx in recommended_arms]

    # Mapear news_id para títulos
    recommended_titles = [news_id_to_title.get(news_id, "Título não encontrado") 
                          for news_id in recommended_news_ids]

    return recommended_titles


def main():
    file_names_users = [f'treino_parte{i}.csv' for i in range(1, 6)]
    file_names_news = [f'itens-parte{i}.csv' for i in range(1, 3)]

    df_users = import_data(USERS_FILE_PATH, file_names_users)
    df_news = import_data(NEWS_FILE_PATH, file_names_news)

    # Juntar os dataframes
    df_joined = join_data(df_users, df_news)

    # Carregar modelo e mapeamento
    mapping_index, mab = load_pkl(MAPPING_INDEX_PATH, MAB_PATH)

    user_id = "2c1080975e257ed630e26679edbe4d5c850c65f3e09f655798b0bba9b42f2110"

    # Verificar se o user_id existe no df_users original
    user_exists = df_users['userId'].eq(user_id).any()

    if not user_exists:
        print(f"Usuário com ID {user_id} não encontrado.")
        return

    # Obter o histórico do usuário do df_users (não do df_joined)
    user_history_str = df_users[df_users['userId'] == user_id]['history'].iloc[0]
    history_ids = user_history_str.split(', ')

    # Obter os títulos das notícias do histórico usando o df_joined
    user_data = df_joined[df_joined['userId'] == user_id]

    if not user_data.empty:
        history_titles = user_data['title'].tolist()
    else:
        # Se o merge falhou, usar o dicionário como fallback
        news_id_to_title = df_news.set_index('page')['title'].to_dict()
        history_titles = [news_id_to_title.get(news_id, "Título não encontrado")
                          for news_id in history_ids]

    # Gerar recomendações
    news_id_to_title = df_news.set_index('page')['title'].to_dict()
    recomendacoes = recommend_news(
        history_ids,
        mab,
        mapping_index,
        news_id_to_title
    )

    print("\nHistórico real do usuário:")
    for titulo in history_titles:
        print(f"- {titulo}")

    print("\nNotícias recomendadas:")
    for titulo in recomendacoes:
        print(f"- {titulo}")


if __name__ == "__main__":
    main()
