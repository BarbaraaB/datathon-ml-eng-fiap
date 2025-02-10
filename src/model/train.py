import os
import sys
import json
import pickle
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.utils import import_data
from src.model.MultiArmedBandit import MultiArmedBandit


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USERS_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'treino')
UTILS_PATH = os.path.join(PROJECT_ROOT, 'src', 'utils')
TRAINED_MODELS_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained_models')


def create_mapping(df_users):
    logger.info('Criando mapeamento de índices das notícias...')
    all_news_ids = set()

    for history_str in df_users['history']:
        news_ids = history_str.split(', ')
        all_news_ids.update(news_ids)

    mapping_index = {news_id: idx for idx, news_id in enumerate(all_news_ids)}

    os.makedirs(UTILS_PATH, exist_ok=True)

    with open(UTILS_PATH + '/mapping_index.json', 'w') as f:
        json.dump(mapping_index, f)
    return mapping_index


def train_mab_model(df_users, mapping_index):
    logger.info('Inicializando e treinando o modelo MAB...')

    n_arms = len(mapping_index)
    mab = MultiArmedBandit(n_arms=n_arms)

    for _, row in df_users.iterrows():
        news_ids = row['history'].split(', ')
        clicks = row['numberOfClicksHistory'].split(', ')

        if len(news_ids) != len(clicks):
            logger.warning(f"Tamanho de history e clicks diferente para o usuário {row['userId']}. Pulando linha.")
            continue

        try:
            clicks = list(map(int, clicks))
        except ValueError as e:
            logger.error(f"Erro ao converter clicks: {e}. Pulando linha.")
            continue

        for news_id, click_count in zip(news_ids, clicks):
            arm_index = mapping_index.get(news_id)
            if arm_index is None:
                logger.warning(f"News ID {news_id} não encontrado no mapeamento. Pulando.")
                continue
            mab.update(arm_index, reward=click_count)

    return mab


def main():
    file_names = [f'treino_parte{i}.csv' for i in range(1, 6)]

    df_users = import_data(USERS_FILE_PATH, file_names)

    mapping_index = create_mapping(df_users)
    mab = train_mab_model(df_users, mapping_index)

    with open(TRAINED_MODELS_PATH + '/mab.pkl', 'wb') as f:
        pickle.dump(mab, f)
    logger.info('Modelo MAB treinado e salvo com sucesso.')


if __name__ == "__main__":
    main()
