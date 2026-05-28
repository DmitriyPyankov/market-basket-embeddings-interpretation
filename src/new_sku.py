import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from tqdm import tqdm
import magpie.sql_utils as su
from pprint import pprint
import yaml
tqdm.pandas()

import numpy as np
import pandas as pd
import torch
from models import vae_model, easy_model

with open('../config/params.yaml', 'r') as file:
    params = yaml.safe_load(file)
with open('./sql_scripts.yaml', 'r') as file:
    sql_scripts = yaml.safe_load(file)

from data_utils import first_prepair, clean_text_light, build_multiview_from_items

def get_random_sku(df_products: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Выбирает случайный SKU, отсутствующий в df_products, и возвращает чеки с этим товаром.

    Алгоритм:
    1. Получает все доступные SKU из внешнего источника.
    2. Отфильтровывает SKU, которые уже присутствуют в df_products.
    3. Случайно выбирает один из новых SKU, обрезает последние 2 символа (помогает избежать часть ошибок в sql запросах).
    4. Ищет до 1500 чеков за фиксированную дату (2026-03-03), содержащих этот SKU.
    5. Если количество уникальных чеков > 100, возвращает DataFrame с чеками и имя SKU.
    6. При неудаче (нет доступных SKU, нет чеков, мало чеков или любое исключение)
       цикл повторяется.

    Returns:
        Tuple[pd.DataFrame, str]: DataFrame с позициями чеков и имя выбранного SKU.
    """
    datetime = params['run_params']['datetime_cheque']
    limit_cheque = params['run_params']['limit_cheque']
    while True:
        try:
            # все sku компании на данный момент
            art_ext = su.get_data_gp(sql_scripts['GET_ALL_SKU'])

            # доступные новые sku (которых ещё нет в df_products)
            available_skus = art_ext[~art_ext.name.isin(df_products.name.tolist())]
            if available_skus.empty:
                continue  # нет новых sku — пробуем снова

            # берём случайный sku
            new_sku = available_skus.sample(n=1)

            new_sku_str_cutted = new_sku['name'].tolist()[0][:-2]
            print(new_sku_str_cutted)

            # cheque_id, где есть этот товар
            prepared_cheque_id = su.get_data_gp(sql_scripts['GET_CHEQUE_ID'].format(
                datetime=datetime,
                sku_name=new_sku_str_cutted,
                limit_cheque=limit_cheque
            ))
            if prepared_cheque_id.empty:
                continue

            list_cheque = ', '.join(prepared_cheque_id.cheque_id.astype(str).tolist())
            # все позиции по найденным чекам
            cheque = su.get_data_gp(sql_scripts['GET_CHEQUE'].format(
                datetime=datetime,
                list_cheque_id=list_cheque
            ))

            # проверяем, что получили достаточно уникальных чеков
            if cheque.empty or cheque.cheque_id.nunique() <= 100:
                continue

            print(cheque.shape[0])
            return cheque, new_sku["name"].iloc[0]

        except Exception:
            continue

def explain_sku(
    df_products: pd.DataFrame,
    df_cheques: pd.DataFrame,
    cheque: pd.DataFrame,
    new_sku_name: str
) -> Dict[str, Any]:
    """
    Строит признаки для нового SKU на основе моделей EASE и VAE.

    Функция выполняет полный pipeline обработки нового товара:
    - очищает входные данные чеков
    - сопоставляет товары с обученным пространством EASE
    - вычисляет вектор совместного потребления (co-occurrence)
    - строит поведенческий профиль товара (EASE)
    - рассчитывает латентное представление через VAE
    В результате новый SKU представляется в том же пространстве,
    что и обученные товары, без переобучения модели.
    Args:
        df_products (pd.DataFrame):
            датасет товаров, использовавшийся при обучении
        df_cheques (pd.DataFrame):
            датасет чеков, использовавшийся при обучении
        cheque (pd.DataFrame):
            новые чеки, содержащие интересующий SKU
        new_sku_name (str):
            название нового товара
    Returns:
        Dict[str, Any]:
            словарь с признаками товара:
            {
                "sku_name": str,
                    исходное название товара
                "sku_clean": str,
                    очищенное название
                "n_cheques": int,
                    количество чеков, в которых встречается товар
                "ease": {
                    "vector": np.ndarray,
                        вектор связей товара со всеми товарами (EASE)
                    "idx_to_product": Dict[int, str],
                        mapping индекса в векторе - название товара
                    "top_id": List[Dict],
                        топ наиболее связанных товаров с весами
                },
                "vae": {
                    "embedding": np.ndarray
                        латентное представление товара (VAE)
                }
            }
    Notes:
        Полученные признаки могут использоваться в downstream задачах:
        - рекомендациях
        - uplift-моделях
        - классификации
        - аналитике
        Функция не выполняет переобучение моделей и работает
        только через инференс и вычисление co-occurrence.
    """

    # а теперь как в EDA уберем чеки с большим кол-вом позиций и всего одной позицией, а так же уберем все пакеты

    bad_treshold = cheque.groupby('cheque_id')['quantity'].count().reset_index().sort_values('quantity')
    bad_treshold_lst = bad_treshold[(bad_treshold.quantity==1) | (bad_treshold.quantity>100)]['cheque_id'].tolist()

    cheque = cheque[~cheque.cheque_id.isin(bad_treshold_lst)]

    mask_paket = cheque['name'].str.contains('Пакет-майка', case=False, na=False)

    cheque = cheque[~mask_paket].reset_index(drop=True)

    # Применяем очистку к name
    cheque['name_clean'] = cheque['name'].progress_apply(clean_text_light)

    del cheque['name']

    df_cheques = pd.read_csv('../data/temp_data_cheque_50.csv')
    df_products = pd.read_csv('../data/temp_data_art_name_50.csv')

    # прогоним обе модели
    vae_res = vae_model(df_products=df_products, df_cheques=df_cheques)
    easy_res = easy_model(df_products=df_products, df_cheques=df_cheques)

    # загрузим полный датасет
    df_old = first_prepair(df_products=df_products, df_cheques=df_cheques)


    HIER_COLS = [
            'art_pricing_model',  # Макро: модель ценообразования
            'art_grp_lvl_0_name',  # Макро-категория
            'art_grp_lvl_1_name',  # Подкатегория
            'art_grp_lvl_2_name'  # Детальная (опционально) особо качества нигде не прибавляет, но всё равно оставим
        ]

    # Уникальные товары по name потому что clean может склеить разные товары в один
    unique_products = df_old[['name', 'name_clean'] + HIER_COLS + ['art_grp_full_name']].drop_duplicates(subset=['name'])

    # Создаём ID для каждого уникального товара
    unique_products = unique_products.reset_index(drop=True)
    unique_products['product_id'] = unique_products.index

    # Маппинг: name - product_id
    name_to_id = dict(zip(unique_products['name'], unique_products['product_id']))
    # Добавляем product_id в основной датафрейм
    df_old['product_id'] = df_old['name'].map(name_to_id)

    # EASY
    B = easy_res['B_matrix']

    product_to_idx = easy_res['product_to_idx']

    idx_to_product = easy_res['idx_to_product']

    print(f"EASE B shape: {B.shape}")

    # Mapping name_clean - product_id
    old_name_to_pid = (
        df_old[["name_clean", "product_id"]]
        .drop_duplicates(subset=["name_clean"])
        .set_index("name_clean")["product_id"]
        .to_dict()
    )

    # Используем  name_clean
    # сопоставляем со старым product_id
    cheque["old_product_id"] = cheque["name_clean"].map(old_name_to_pid)

    # сопоставляем с EASE индексом
    cheque["ease_idx"] = cheque["old_product_id"].map(product_to_idx)

    # Проверка покрытия
    total_rows = len(cheque)
    matched_rows = cheque["ease_idx"].notna().sum()

    print("\nПокрытие:")
    print(f"Всего строк:   {total_rows}")
    print(f"Сопоставлено:  {matched_rows}")
    print(f"Доля:          {matched_rows / total_rows:.2%}")

    print("\nЧеков:", cheque["cheque_id"].nunique())

    # Диагностика
    print("\nНЕ сопоставились (пример):")
    pprint(
        cheque[cheque["ease_idx"].isna()][["name_clean"]]
        .drop_duplicates()
        .head(10)
    )

    print("\nСопоставились (пример):")
    pprint(
        cheque[cheque["ease_idx"].notna()][["name_clean", "old_product_id", "ease_idx"]]
        .drop_duplicates()
        .head(10)
    )

    # определяем новый SKU
    new_name_clean = clean_text_light(new_sku_name)

    print("Новый SKU:", new_name_clean)

    # чеки, где он есть
    new_sku_cheques = cheque[cheque['name_clean'] == new_name_clean]['cheque_id'].unique()

    print("Чеков с новым SKU:", len(new_sku_cheques))

    # берём только строки этих чеков
    subset = cheque[cheque['cheque_id'].isin(new_sku_cheques)].copy()

    print("Строк в subset:", len(subset))

    # оставляем только известные товары
    subset_known = subset[subset['ease_idx'].notna()].copy()

    print("Известных товаров:", len(subset_known))

    # считаем co-occurrence
    co_counts = subset_known.groupby('ease_idx')['cheque_id'].nunique()

    print("\nТоп co-occurrence:")
    pprint(co_counts.sort_values(ascending=False).head(10))

    # нормализация
    co_vector = np.zeros(len(product_to_idx), dtype=np.float32)

    for idx, val in co_counts.items():
        co_vector[int(idx)] = val

    co_vector = co_vector / (co_vector.sum() + 1e-8)

    print("Вектор готов:", co_vector.shape)

    b_new = co_vector.copy()

    # нормализуем как строки B
    b_new = b_new / (np.linalg.norm(b_new) + 1e-8)

    print("b_new shape:", b_new.shape)

    n_cheques = len(new_sku_cheques)

    # EASE
    id_to_name = dict(zip(df_old['product_id'], df_old['name_clean']))

    top_idx = np.argsort(-b_new)[:10]

    ease_top = [
        {"product_id": int(idx_to_product[i]),
            "name": id_to_name.get(idx_to_product[i], str(idx_to_product[i])),
            "weight": float(b_new[i])
        } for i in top_idx ]


    # Это и есть наш выход для easy
    ease_features = {
        "vector": b_new,
        "idx_to_product": id_to_name,
        "top_id": ease_top
    }

    # VAE
    # берём только нужные чеки
    subset = cheque[cheque['cheque_id'].isin(new_sku_cheques)].copy()

    # группируем в корзины
    baskets = subset.groupby('cheque_id')['name_clean'].apply(list).tolist()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vae_res['model']
    item_to_idx_vae = vae_res['item_to_idx']
    lvl2_to_idx = vae_res['lvl2_to_idx']
    lvl1_to_idx = vae_res['lvl1_to_idx']
    lvl0_to_idx = vae_res['lvl0_to_idx']
    idx_to_item = vae_res['idx_to_item']

    print("BUILD CATEGORY MAPPINGS")

    meta = df_old[[
        'name_clean',
        'art_grp_lvl_2_name',
        'art_grp_lvl_1_name',
        'art_grp_lvl_0_name'
    ]].drop_duplicates()

    # mapping name - категории
    name_to_lvl2 = dict(zip(meta['name_clean'], meta['art_grp_lvl_2_name']))
    name_to_lvl1 = dict(zip(meta['name_clean'], meta['art_grp_lvl_1_name']))
    name_to_lvl0 = dict(zip(meta['name_clean'], meta['art_grp_lvl_0_name']))

    build_multiview_from_items_dict = {
                'item_to_idx': item_to_idx_vae, 'lvl2_to_idx': lvl2_to_idx, 
                'lvl1_to_idx': lvl1_to_idx, 'lvl0_to_idx': lvl0_to_idx,
                'item_to_lvl2': name_to_lvl2, 'item_to_lvl1': name_to_lvl1, 
                'item_to_lvl0': name_to_lvl0, 'device': device
            }

    # VAE
    embeddings = []

    for basket in baskets:
        x_item, x_lvl2, x_lvl1, x_lvl0 = build_multiview_from_items(basket, build_multiview_from_items_dict)

        x = {
            'item': torch.FloatTensor(x_item).unsqueeze(0).to(device),
            'lvl2': torch.FloatTensor(x_lvl2).unsqueeze(0).to(device),
            'lvl1': torch.FloatTensor(x_lvl1).unsqueeze(0).to(device),
            'lvl0': torch.FloatTensor(x_lvl0).unsqueeze(0).to(device)
        }

        with torch.no_grad():
            mu, _ = model.encode(x)

        embeddings.append(mu.cpu().numpy().flatten())

    z_new = np.mean(embeddings, axis=0)

    vae_features = {
        "embedding": z_new
    }

    features = {
        "sku_name": new_sku_name,
        "sku_clean": new_name_clean,
        "n_cheques": n_cheques,
        "ease": ease_features,
        "vae": vae_features
    }
    return features



if __name__ == "__main__":
    # Загрузка данных, использованных при обучении моделей
    df_cheques = pd.read_csv('../data/temp_data_cheque_50.csv')
    df_products = pd.read_csv('../data/temp_data_art_name_50.csv')

    # Получение случайного нового SKU и чеков с ним
    cheque, new_sku_name = get_random_sku(df_products=df_products)

    # Построение признаков для нового SKU (EASE + VAE)
    features = explain_sku(
        df_products=df_products,
        df_cheques=df_cheques,
        cheque=cheque,
        new_sku_name=new_sku_name
    )

    # Вывод результата
    pprint(features)