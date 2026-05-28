import pandas as pd
from typing import Union, List, Any, Tuple
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
from tqdm import tqdm
from collections import Counter
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import build_multiview_from_items

with open('../config/params.yaml', 'r') as file:
    params = yaml.safe_load(file)

RECALL_AT_k = params['run_params']['recall_at_k']
SEED = params['run_params']['STATE']
SAMPLES = params['run_params']['n_samples']

def recall_at_k_random_holdout_word2vec(
    model: Word2Vec,
    test_baskets: List[List[str]],
    k: int = RECALL_AT_k,
    random_state: int = SEED
) -> float:
    """
    Считает Recall@K для Word2Vec по схеме leave-one-out.
    В каждой корзине случайный товар скрывается, остальные используются
    как контекст. Модель должна угадать скрытый товар.
    Returns:
        float: значение Recall@K
    """
    rng = np.random.default_rng(random_state)
    recalls = []

    for basket in tqdm(test_baskets):
        basket = list(basket)

        # убираем дубли внутри корзины для честной multi-hot логики
        basket = list(dict.fromkeys(basket))

        if len(basket) < 2:
            continue

        target = str(rng.choice(basket))
        context = [str(x) for x in basket if str(x) != target]

        if len(context) == 0:
            continue

        scores = {}

        for item in model.wv.index_to_key:
            if item in context:
                continue

            score = 0.0
            for c in context:
                if c in model.wv:
                    score += model.wv.similarity(item, c)

            scores[item] = score

        top_k = sorted(scores, key=scores.get, reverse=True)[:k]
        recalls.append(int(target in top_k))

    return sum(recalls) / len(recalls) if recalls else 0.0 

def popularity_recall_random_holdout_word2vec(
    test_baskets: List[List[int]],
    baskets: List[List[int]],
    k: int = RECALL_AT_k,
    random_state: int = SEED
) -> float:
    """
    Считает Recall@K для популярностного baseline.
    В качестве рекомендаций используются самые частые товары.
    """
    rng = np.random.default_rng(random_state)
    recalls = []

    for basket in tqdm(test_baskets):
        basket = list(basket)

        # убираем дубли
        basket = list(dict.fromkeys(basket))

        if len(basket) < 2:
            continue

        target = rng.choice(basket)
        context = [x for x in basket if x != target]

        if len(context) == 0:
            continue

        # исключаем уже увиденные товары из рекомендаций
        recs = [x for x in _popularity_recommend(baskets, k=1000) if x not in context][:k]

        recalls.append(int(target in recs))

    return sum(recalls) / len(recalls) if recalls else 0.0

    
def _popularity_recommend(
        baskets: List[List[int]], 
        k: int = RECALL_AT_k
    ) -> List[int]:
    """
    Возвращает топ-k самых популярных товаров по всем корзинам.
    """
    item_counts = Counter([item for basket in baskets for item in basket])
    top_items = [item for item, _ in item_counts.most_common(50)]
    return top_items[:k]

def recall_vae(
    model: Any,
    baskets: pd.DataFrame,
    build_multiview_from_items_dict: dict,
    k: int = RECALL_AT_k,
    n: int = SAMPLES,
    seed: int = SEED
) -> float:
    """
    Считает Recall@K для VAE модели.
    Для каждой корзины случайный товар скрывается,
    а модель должна его восстановить.
    """

    device = build_multiview_from_items_dict['device']
    item_to_idx = build_multiview_from_items_dict['item_to_idx']

    rng = np.random.default_rng(seed)
    
    indices = rng.choice(len(baskets), size=n, replace=False)
    
    hits = 0
    
    for i in tqdm(indices):
        items = list(baskets.iloc[i]['name_clean'])
        items = list(dict.fromkeys(items))
        
        if len(items) < 2:
            continue
        
        target = rng.choice(items)
        context = [x for x in items if x != target]
        
        x = build_multiview_from_items(item_names=context, build_multiview_from_items_dict=build_multiview_from_items_dict)
        
        x = {
            'item': torch.FloatTensor(x[0]).unsqueeze(0).to(device),
            'lvl2': torch.FloatTensor(x[1]).unsqueeze(0).to(device),
            'lvl1': torch.FloatTensor(x[2]).unsqueeze(0).to(device),
            'lvl0': torch.FloatTensor(x[3]).unsqueeze(0).to(device)
        }
        
        with torch.no_grad():
            recon, _, _ = model(x)
            scores = recon['item'].cpu().numpy().flatten()
        
        context_idx = [item_to_idx[x] for x in context if x in item_to_idx]
        scores[context_idx] = -np.inf
        
        top = np.argpartition(-scores, k)[:k]
        
        if target in item_to_idx and item_to_idx[target] in top:
            hits += 1
    
    return hits / n

def loss_fn(
    recon: dict,
    x: dict,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Функция потерь для VAE.
    Состоит из:
    - BCE для всех уровней (item + категории)
    - KL-дивергенции
    Returns:
        total, bce, kl
    """

    bce_item = F.binary_cross_entropy_with_logits(
        recon['item'], x['item'], reduction='sum'
    )
    bce_lvl2 = F.binary_cross_entropy_with_logits(
        recon['lvl2'], x['lvl2'], reduction='sum'
    )
    bce_lvl1 = F.binary_cross_entropy_with_logits(
        recon['lvl1'], x['lvl1'], reduction='sum'
    )
    bce_lvl0 = F.binary_cross_entropy_with_logits(
        recon['lvl0'], x['lvl0'], reduction='sum'
    )

    bce = (
        0.7 * bce_item +
        1.0 * bce_lvl2 +
        0.8 * bce_lvl1 +
        0.6 * bce_lvl0
    )
    
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = bce + beta * kl
    
    return total, bce, kl

def recall_ease(
    B: np.ndarray,
    X_test: csr_matrix,
    k: int = 10,
    n_samples: int = SAMPLES,
    seed: int = SEED
) -> float:
    """
    Считает Recall@K для модели EASE.
    Использует сумму весов связей между товарами в корзине.
    """
    rng = np.random.default_rng(seed)
    
    indices = rng.choice(X_test.shape[0], size=n_samples, replace=False)
    
    hits = 0
    total = 0
    
    for i in tqdm(indices):
        row = X_test[i].toarray().flatten()
        
        items = np.where(row == 1)[0]
        
        if len(items) < 2:
            continue
        
        target = rng.choice(items)
        context = [x for x in items if x != target]
        
        if len(context) == 0:
            continue
        
        scores = np.zeros(B.shape[0])
        
        for c in context:
            scores += B[c]
        
        scores[context] = -np.inf
        
        top_k = np.argpartition(-scores, k)[:k]
        
        if target in top_k:
            hits += 1
        
        total += 1
    
    return hits / total if total > 0 else 0.0

def vae_scores_in_ease_space(
    context: List[int],
    build_multiview_from_items_dict: dict,
    idx_to_product: dict,
    pid_to_name: dict,
    model: Any,
    ease_to_vae: dict
) -> np.ndarray:
    """
    Получает скоринг VAE и переводит его в пространство EASE.
    На выходе вектор размера n_items с оценками для всех товаров.
    """
    device = build_multiview_from_items_dict['device']
    context_names = []
    for ease_idx in context:
        pid = idx_to_product[ease_idx]
        name = pid_to_name.get(pid)
        if name is not None:
            context_names.append(name)

    x = build_multiview_from_items(item_names=context_names, build_multiview_from_items_dict=build_multiview_from_items_dict)
    x = {
        'item': torch.FloatTensor(x[0]).unsqueeze(0).to(device),
        'lvl2': torch.FloatTensor(x[1]).unsqueeze(0).to(device),
        'lvl1': torch.FloatTensor(x[2]).unsqueeze(0).to(device),
        'lvl0': torch.FloatTensor(x[3]).unsqueeze(0).to(device)
    }

    with torch.no_grad():
        recon, _, _ = model(x, sample=False)
        raw_scores = recon['item'].cpu().numpy().flatten()

    scores = np.zeros(len(idx_to_product), dtype=np.float32)
    for ease_idx, vae_idx in ease_to_vae.items():
        scores[ease_idx] = raw_scores[vae_idx]

    return scores

def recall_hybrid_model(X_test, hybrid_fn, n_samples=SAMPLES, k=RECALL_AT_k, seed=SEED):
    """
    Оценивает качество гибридной модели по метрике Recall@K.
    Для каждой тестовой корзины случайным образом выбирается один товар,
    который считается целевым (target). Остальные товары формируют контекст.
    Модель должна предсказать целевой товар, используя только контекст.
    Процесс:
    1. Выбор случайной корзины
    2. Скрытие одного товара (target)
    3. Прогон контекста через модель (hybrid_fn)
    4. Проверка, попал ли target в top-K

    Args:
        X_test (csr_matrix): матрица корзин (cheque_id × product_id)
        hybrid_fn (callable): функция предсказания (например hybrid_scores)
        n_samples (int): количество случайных корзин для оценки
        k (int): размер top-K рекомендаций
        seed (int): random seed

    Returns:
        float: значение Recall@K
    """
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(X_test.shape[0], size=n_samples, replace=False)

    hits = 0
    total = 0

    for i in tqdm(indices):
        row = X_test[i].toarray().flatten()
        items = np.where(row == 1)[0]

        if len(items) < 2:
            continue

        target = rng.choice(items)
        context = [x for x in items if x != target]

        candidates, scores = hybrid_fn(context)

        top_k = candidates[np.argsort(-scores)[:k]]

        if target in top_k:
            hits += 1
        
        total += 1

    return hits / total if total > 0 else 0.0  