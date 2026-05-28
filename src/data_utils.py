import pandas as pd
import re
import numpy as np
from typing import Union, List, Dict, Tuple
import torch

def clean_text_light(text: Union[str, float, None]) -> str:
    """
    Лёгкая очистка текста для товарных наименований.

    Выполняет минимальную нормализацию: приводит к нижнему регистру,
    схлопывает пробелы, удаляет большинство спецсимволов, но сохраняет
    буквы, цифры, пробелы и дефисы. Полезно для подготовки названий,
    где важны бренды, артикулы и ключевые характеристики (без агрессивного стемминга).

    Returns
    str
        Очищенная строка в нижнем регистре без лишних пробелов и специальных символов.
    """

    if pd.isna(text):
        return ""
    
    # Приводим к нижнему регистру
    text = str(text).lower()
    
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Убираем спецсимволы, но оставляем цифры, буквы, пробелы, дефисы
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # Убираем множественные дефисы
    text = re.sub(r'-+', '-', text)
    
    # Trim
    text = text.strip()
    
    return text

def first_prepair(df_products: pd.DataFrame, df_cheques: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает данные, объединяя товары и чеки, очищает названия,
    удаляет чеки с некорректным количеством позиций (1 или >100) и исключает пакеты-майки.

    Шаги:
    1. Объединение df_products и df_cheques по cheque_id (left join).
    2. Очистка названия товара (name) через clean_text_light.
    3. Вычисление количества позиций на чек, удаление чеков с количеством == 1 или >100.
    4. Удаление строк, где name содержит 'Пакет-майка'
    5. Сброс индекса.
    Parameters
    df_products : pd.DataFrame
        Датафрейм с товарами, должен содержать колонки 'cheque_id', 'name', 'quantity'.
    df_cheques : pd.DataFrame
        Датафрейм с чеками, должен содержать колонку 'cheque_id'.
    Returns
    pd.DataFrame
    """

    df = df_products.merge(df_cheques, on='cheque_id', how='left')
    # Применяем очистку к name
    df['name_clean'] = df['name'].progress_apply(clean_text_light)
    bad_treshold = df.groupby('cheque_id')['quantity'].count().reset_index().sort_values('quantity')
    bad_treshold_lst = bad_treshold[(bad_treshold.quantity==1) | (bad_treshold.quantity>100)]['cheque_id'].tolist()
    df = df[~df.cheque_id.isin(bad_treshold_lst)]
    mask_paket = df['name'].str.contains('Пакет-майка', case=False, na=False)
    df = df[~mask_paket].reset_index(drop=True)

    return df

def build_vector(elements: List[str], mapping: Dict[str, int]) -> np.ndarray:
    """
    Строит multi-hot вектор по списку элементов.
    Каждый элемент из списка отображается в индекс через mapping.
    Если элемент присутствует, соответствующая позиция вектора равна 1.
    Args:
        elements: список объектов (например, товаров или категорий)
        mapping: словарь элемент - индекс
    Returns:
        np.ndarray: multi-hot вектор фиксированной длины
    """
    # функции создания multi-hot
    vec = np.zeros(len(mapping), dtype=np.float32)
    for e in elements:
        if e in mapping:
            vec[mapping[e]] = 1.0
    return vec

def build_multiview_from_items(
    item_names: List[str],
    build_multiview_from_items_dict: Dict[str, Dict]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Формирует multi-view представление корзины для VAE.
    Для списка товаров строит четыре вектора:
    - item уровень
    - lvl2
    - lvl1
    - lvl0
    Использует маппинги товаров и категорий.
    Args:
        item_names: список названий товаров
        build_multiview_from_items_dict: словарь с маппингами
    Returns:
        tuple:
            x_item, x_lvl2, x_lvl1, x_lvl0 — multi-hot векторы
    """
    item_to_idx = build_multiview_from_items_dict['item_to_idx']
    lvl1_to_idx = build_multiview_from_items_dict['lvl1_to_idx']
    lvl0_to_idx = build_multiview_from_items_dict['lvl0_to_idx']
    item_to_lvl2 = build_multiview_from_items_dict['item_to_lvl2']
    item_to_lvl1 = build_multiview_from_items_dict['item_to_lvl1']
    item_to_lvl0 = build_multiview_from_items_dict['item_to_lvl0']
    lvl2_to_idx = build_multiview_from_items_dict['lvl2_to_idx']

    x_item = np.zeros(len(item_to_idx), dtype=np.float32)
    x_lvl2 = np.zeros(len(lvl2_to_idx), dtype=np.float32)
    x_lvl1 = np.zeros(len(lvl1_to_idx), dtype=np.float32)
    x_lvl0 = np.zeros(len(lvl0_to_idx), dtype=np.float32)
    
    for item in item_names:
        if item in item_to_idx:
            x_item[item_to_idx[item]] = 1.0
        
        if item in item_to_lvl2 and item_to_lvl2[item] in lvl2_to_idx:
            x_lvl2[lvl2_to_idx[item_to_lvl2[item]]] = 1.0
        
        if item in item_to_lvl1 and item_to_lvl1[item] in lvl1_to_idx:
            x_lvl1[lvl1_to_idx[item_to_lvl1[item]]] = 1.0
        
        if item in item_to_lvl0 and item_to_lvl0[item] in lvl0_to_idx:
            x_lvl0[lvl0_to_idx[item_to_lvl0[item]]] = 1.0
    
    return x_item, x_lvl2, x_lvl1, x_lvl0

def _mask_numpy_row(x: np.ndarray, p: float) -> np.ndarray:
    """
    Случайно зануляет часть активных элементов в векторе.
    Используется как data augmentation для VAE.
    Args:
        x: бинарный вектор
        p: доля зануляемых элементов
    Returns:
        np.ndarray: замаскированный вектор
    """
    x = x.copy()
    idx = np.where(x == 1)[0]
    if len(idx) > 1:
        n_drop = max(1, int(len(idx) * p))
        drop = np.random.choice(idx, n_drop, replace=False)
        x[drop] = 0
    return x

def apply_train_mask(item_tensor: torch.Tensor, p: float) -> torch.Tensor:
    """
    Применяет случайную маску к батчу товаров.
    Для каждой строки зануляет часть активных элементов.
    Args:
        item_tensor: тензор (batch_size, n_items)
        p: доля маскирования
    Returns:
        torch.Tensor: замаскированный тензор
    """
    masked_np = np.array(
        [_mask_numpy_row(x, p=p) for x in item_tensor.cpu().numpy()],
        dtype=np.float32
    )
    return torch.from_numpy(masked_np).to(item_tensor.device)

def basket_emb_vae(
    context: List[int],
    ease_to_vae: Dict[int, int],
    vae_item_emb: np.ndarray
) -> np.ndarray:
    """
    Строит embedding корзины как среднее embedding товаров.
    Использует mapping EASE - VAE для согласования пространств.
    Args:
        context: список индексов товаров (EASE)
        ease_to_vae: mapping индексов EASE -> VAE
        vae_item_emb: матрица embedding товаров
    Returns:
        np.ndarray: нормализованный embedding корзины
    """
    vecs = []
    for ease_idx in context:
        if ease_idx in ease_to_vae:
            vecs.append(vae_item_emb[ease_to_vae[ease_idx]])
    if not vecs:
        return np.zeros(vae_item_emb.shape[1], dtype=np.float32)

    emb = np.mean(vecs, axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb.astype(np.float32)