import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.sparse import csr_matrix
from tqdm import tqdm
tqdm.pandas()
import torch
import yaml

from data_utils import basket_emb_vae
from metrics import vae_scores_in_ease_space 

with open('../config/params.yaml', 'r') as file:
    params = yaml.safe_load(file)

SEED = params['run_params']['STATE']

def build_features(
    ease_scores: np.ndarray,
    vae_scores: np.ndarray,
    candidates: np.ndarray,
    context: List[int],
    ranks: np.ndarray,
    b_emb: np.ndarray,
    ease_to_vae: dict,
    vae_item_emb: np.ndarray
) -> np.ndarray:
    """
    Формирует признаки для кандидатов на основе EASE и VAE.
    Для каждого кандидата рассчитываются:
    - нормализованный EASE скор
    - нормализованный VAE скор
    - косинусная близость embedding корзины и товара
    - взаимодействия признаков
    - размер корзины
    - позиция кандидата в ранжировании EASE
    Args:
        ease_scores: вектор скорингов EASE по всем товарам
        vae_scores: вектор скорингов VAE по всем товарам
        candidates: индексы кандидатов
        context: список товаров в корзине
        ranks: позиции кандидатов по EASE
        b_emb: embedding корзины (VAE)
        ease_to_vae: mapping EASE - VAE
        vae_item_emb: embedding товаров
    Returns:
        np.ndarray:
            матрица признаков размера (n_candidates, n_features)
    """
    e = ease_scores[candidates].astype(np.float32)
    v = vae_scores[candidates].astype(np.float32)

    # z-score внутри candidate set
    e = (e - e.mean()) / (e.std() + 1e-8)
    v = (v - v.mean()) / (v.std() + 1e-8)

    feats = []

    for i, cand in enumerate(candidates):
        if cand in ease_to_vae:
            cand_emb = vae_item_emb[ease_to_vae[cand]]
            cos_vae = float(np.dot(b_emb, cand_emb))
        else:
            cos_vae = 0.0

        feats.append([
            e[i],     # 0: normalized ease
            v[i],     # 1: normalized vae score
            cos_vae,     # 2: basket-item cosine in VAE space
            e[i] * v[i],     
            e[i] * cos_vae,     
            v[i] * cos_vae,     
            float(len(context)),     
            float(ranks[i]),     
        ])

    return np.array(feats, dtype=np.float32)

def build_ranking_dataset(
    X_source: csr_matrix,
    ease_to_vae: dict,
    vae_item_emb: np.ndarray,
    build_multiview_from_items_dict: dict,
    B: np.ndarray,
    idx_to_product: dict,
    pid_to_name: dict,
    model,
    n_samples: int,
    topk_candidates: int,
    seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Строит датасет для обучения модели ранжирования (reranker).
    Для каждой корзины:
    - случайно скрывается один товар (target)
    - формируется контекст
    - с помощью EASE выбираются кандидаты
    - для кандидатов считаются признаки (EASE + VAE)
    - формируются пары (признаки, метка)
    Args:
        X_source: матрица корзин (cheque × товары)
        ease_to_vae: mapping EASE - VAE
        vae_item_emb: embedding товаров
        build_multiview_from_items_dict: словарь для VAE
        B: матрица EASE
        idx_to_product: mapping индекс - product_id
        pid_to_name: mapping product_id - name
        model: обученная VAE модель
        n_samples: количество корзин для обучения
        topk_candidates: количество кандидатов
        seed: random seed
    Returns:
        tuple:
            X_feat (np.ndarray):
                матрица признаков
            y (np.ndarray):
                метки (1 — target, 0 — нет)
            group (List[int]):
                размеры групп (для ранжирования)
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(X_source.shape[0], size=min(n_samples, X_source.shape[0]), replace=False)

    X_feat = []
    y = []
    group = []

    for i in tqdm(indices):
        row = X_source[i].toarray().flatten()
        items = np.where(row == 1)[0]

        if len(items) < 2:
            continue

        target = rng.choice(items)
        context = [x for x in items if x != target]

        if not context:
            continue

        ease_scores = np.zeros(B.shape[0], dtype=np.float32)
        for c in context:
            ease_scores += B[c]
        ease_scores[context] = -np.inf

        candidates = np.argpartition(-ease_scores, topk_candidates)[:topk_candidates]

        if target not in candidates:
            candidates = np.append(candidates[:-1], target)

        vae_scores = vae_scores_in_ease_space(context, build_multiview_from_items_dict, 
                                              idx_to_product, pid_to_name, model, ease_to_vae)
        b_emb = basket_emb_vae(context, ease_to_vae, vae_item_emb)

        cand_scores = ease_scores[candidates]
        order = np.argsort(-cand_scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(candidates))

        X_local = build_features(ease_scores, vae_scores, candidates, context, ranks, b_emb, ease_to_vae, vae_item_emb)

        for j, cand in enumerate(candidates):
            X_feat.append(X_local[j])
            y.append(int(cand == target))

        group.append(len(candidates))

    return np.array(X_feat, dtype=np.float32), np.array(y, dtype=np.int32), group

def hybrid_scores(
    context: List[int],
    build_multiview_from_items_dict: Dict[str, Any],
    ease_to_vae: Dict[int, int],
    vae_item_emb: np.ndarray,
    B: np.ndarray,
    idx_to_product: Dict[int, int],
    pid_to_name: Dict[int, str],
    model: torch.nn.Module,
    scaler: Any,
    reranker: Any,
    topk_candidates: int,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет финальные скоры товаров для заданного контекста корзины
    с использованием двухэтапной гибридной модели (EASE + VAE + reranker).
    Алгоритм работы:
    1. Генерация кандидатов:
       - для заданного контекста (корзины) вычисляются скоры EASE
       - выбираются top-K товаров (candidate generation)
    2. Формирование признаков:
       - для каждого кандидата рассчитываются:
         • EASE score (сумма связей с товарами контекста)
         • VAE score (оценка реконструкции)
         • косинусная близость embedding корзины и товара (VAE)
         • позиция кандидата в ранжировании EASE
         • взаимодействия признаков (feature interactions)
    3. Ранжирование:
       - признаки нормализуются и подаются в обученный reranker
         (логистическая регрессия)
       - получаются вероятности релевантности кандидатов
    4. Финальный скор:
       - нормализованные скоры EASE и reranker объединяются:
         final_score = EASE_norm + alpha * reranker_norm
    Args:
        context (list[int]):
            список индексов товаров (product_to_idx), формирующих корзину
        build_multiview_from_items_dict (dict):
            словарь с маппингами для построения входа VAE:
            - item_to_idx
            - lvl2_to_idx, lvl1_to_idx, lvl0_to_idx
            - item_to_lvl2 / lvl1 / lvl0
            - device
        ease_to_vae (dict):
            mapping индексов товаров из пространства EASE
            в индексы пространства VAE
        vae_item_emb (np.ndarray):
            embedding товаров из VAE (обычно веса decoder слоя)
        B (np.ndarray):
            матрица весов EASE (item-item связи)
        idx_to_product (dict):
            mapping индекса - product_id
        pid_to_name (dict):
            mapping product_id - name_clean (для VAE)
        model (torch.nn.Module):
            обученная модель VAE
        scaler (sklearn.preprocessing):
            объект нормализации признаков
        reranker (sklearn model):
            обученная модель ранжирования (логистическая регрессия)
        topk_candidates (int, optional):
            количество кандидатов, отбираемых моделью EASE (по умолчанию 200)
        alpha (float, optional):
            коэффициент влияния reranker на итоговый скор (по умолчанию 0.5)
    Returns:
        tuple:
            candidates (np.ndarray):
                индексы товаров-кандидатов
            final_scores (np.ndarray):
                итоговые скоры кандидатов после гибридного ранжирования
    """

    ease_scores = np.zeros(B.shape[0], dtype=np.float32)
    for c in context:
        ease_scores += B[c]
    ease_scores[context] = -np.inf

    candidates = np.argpartition(-ease_scores, topk_candidates)[:topk_candidates]

    vae_scores = vae_scores_in_ease_space(context, build_multiview_from_items_dict,
                                          idx_to_product, pid_to_name, model, ease_to_vae)
    b_emb = basket_emb_vae(context, ease_to_vae, vae_item_emb)

    cand_scores = ease_scores[candidates]
    order = np.argsort(-cand_scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(candidates))

    X_feat = build_features(ease_scores, vae_scores, candidates, context, ranks, b_emb, ease_to_vae, vae_item_emb)
    X_feat = scaler.transform(X_feat)

    probs = reranker.predict_proba(X_feat)[:, 1]

    e = cand_scores
    e = (e - e.mean()) / (e.std() + 1e-8)

    p = probs
    p = (p - p.mean()) / (p.std() + 1e-8)

    final_scores = e + alpha * p

    return candidates, final_scores

def get_sku_features(
    product_id: int,
    B: np.ndarray,
    product_to_idx: Dict[int, int],
    idx_to_product: Dict[int, int],
    ease_to_vae: Dict[int, int],
    vae_item_emb: np.ndarray,
    topn_ease: int
) -> Optional[Dict[str, object]]:
    """
    Возвращает признаки для заданного SKU на основе моделей EASE и VAE.
    Для товара формируются:
    - поведенческие связи (EASE): топ-N наиболее связанных товаров
    - латентное представление (VAE): embedding товара
    Args:
        product_id (int):
            идентификатор товара
        B (np.ndarray):
            матрица весов EASE (item-item связи)
        product_to_idx (Dict[int, int]):
            mapping product_id - индекс в матрице EASE
        idx_to_product (Dict[int, int]):
            mapping индекс - product_id
        ease_to_vae (Dict[int, int]):
            mapping индексов EASE - индексы VAE
        vae_item_emb (np.ndarray):
            embedding товаров из VAE
        topn_ease (int, optional):
            количество возвращаемых связанных товаров (по умолчанию 10)
    Returns:
        Optional[Dict[str, object]]:
            словарь с признаками товара:
            {
                "product_id": int,
                "ease_top_related": List[Tuple[int, float]],
                "vae_embedding": np.ndarray | None
            }
            Если товар отсутствует в модели EASE -  возвращается None
    """
    if product_id not in product_to_idx:
        return None

    ease_idx = product_to_idx[product_id]

    # EASE
    ease_weights = B[ease_idx]
    top_idx = np.argsort(-ease_weights)[:topn_ease]
    top_related = [(idx_to_product[i], float(ease_weights[i])) for i in top_idx]

    # VAE
    if ease_idx in ease_to_vae:
        vae_idx = ease_to_vae[ease_idx]
        vae_embedding = vae_item_emb[vae_idx]
    else:
        vae_embedding = None

    return {
        "product_id": product_id,
        "ease_top_related": top_related,
        "vae_embedding": vae_embedding
    }