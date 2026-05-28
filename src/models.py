import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Any
import pandas as pd
import random
from collections import defaultdict
from tqdm import tqdm
tqdm.pandas()

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

import copy
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_utils import first_prepair, build_vector, build_multiview_from_items, apply_train_mask
from metrics import (popularity_recall_random_holdout_word2vec, recall_hybrid_model,
                     recall_at_k_random_holdout_word2vec, recall_vae, loss_fn, recall_ease)
from model_utils import hybrid_scores, build_ranking_dataset, get_sku_features
from common import VAE, MultiViewDataset

with open('../config/params.yaml', 'r') as file:
    params = yaml.safe_load(file)

TEST_SIZE=params['run_params']['test_size']

def baseline(df_products: pd.DataFrame, df_cheques: pd.DataFrame) -> pd.DataFrame:
    """
    Обучает baseline модель Word2Vec на корзинах товаров и возвращает embedding товаров.

    Внутри функции:
    - выполняется предобработка данных
    - формируются корзины с учётом количества товаров (quantity)
    - обучается модель Word2Vec
    - извлекаются embedding товаров
    - проводится простая проверка качества (similarity + recall)
    Args:
        df_products (pd.DataFrame):
            таблица товаров
        df_cheques (pd.DataFrame):
            таблица чеков (покупок)
    Returns:
        pd.DataFrame:
            таблица с embedding товаров и метаданными
    """

    df_final = first_prepair(df_products=df_products, df_cheques=df_cheques)

    HIER_COLS = params['run_params']['HIER_COLS']
    SEED = params['run_params']['STATE']

    # Уникальные товары по name потому что clean может склеить разные товары в один
    unique_products = df_final[['name', 'name_clean'] + HIER_COLS + ['art_grp_full_name']].drop_duplicates(subset=['name'])

    # Создаём ID для каждого уникального товара
    unique_products = unique_products.reset_index(drop=True)
    unique_products['product_id'] = unique_products.index

    print(f" Уникальных товаров: {len(unique_products):,}")

    # Маппинг: name - product_id
    name_to_id = dict(zip(unique_products['name'], unique_products['product_id']))

    # Маппинг: product_id - метаданные
    product_metadata = unique_products.set_index('product_id').to_dict('index')

    # Добавляем product_id в основной датафрейм
    df_final['product_id'] = df_final['name'].map(name_to_id)

    # Проверка
    assert df_final['product_id'].isna().sum() == 0, "Есть товары без ID!"

    baskets_dict = defaultdict(list)
    for _, row in df_final.iterrows():
        cheque_id = row['cheque_id']
        product_id = row['product_id']
        quantity = row['quantity']
        
        # Добавляем товар quantity раз (как в Word2Vec для NLP)
        # Если quantity=2, товар появится 2 раза в корзине
        for _ in range(int(quantity)):
            baskets_dict[cheque_id].append(product_id)

    # Конвертируем в список списков (формат для Word2Vec)
    baskets = list(baskets_dict.values())

    print(f" Сформировано {len(baskets):,} корзин")
    print(f"   Средний размер корзины: {np.mean([len(b) for b in baskets]):.2f} товаров")
    print(f"   Медианный размер: {np.median([len(b) for b in baskets]):.0f} товаров")

    train_baskets, test_baskets = train_test_split(
        baskets, 
        test_size=TEST_SIZE, 
        random_state=SEED)

    print(f"   - Уникальных товаров: {len(unique_products):,}")

    print("\n ПОДГОТОВКА ДАННЫХ ДЛЯ WORD2VEC")

    # Word2Vec требует строковые ID
    baskets_str = [[str(pid) for pid in basket] for basket in train_baskets]

    print(f" Преобразовано в строковые ID")
    print(f"   Пример первой корзины (ID): {baskets_str[0][:10]}")


    print("\n ОБУЧЕНИЕ WORD2VEC")

    # Гиперпараметры
    VECTOR_SIZE = params['w2v']['hiperparams']['VECTOR_SIZE']    # размер эмбеддинга (64 измерения)
    WINDOW = params['w2v']['hiperparams']['WINDOW']    # окно контекста (10 товаров вокруг)
    MIN_COUNT = params['w2v']['hiperparams']['MIN_COUNT']    # минимальная частота товара (фильтр редких)
    EPOCHS = params['w2v']['hiperparams']['EPOCHS']    # количество эпох обучения
    WORKERS = params['w2v']['hiperparams']['WORKERS']    # количество потоков
    SG = params['w2v']['hiperparams']['SG']    # Skip-gram (1) или CBOW (0)
    NEGATIVE = params['w2v']['hiperparams']['NEGATIVE']    # negative sampling

    print(f" Гиперпараметры:")
    print(f"   vector_size: {VECTOR_SIZE}")
    print(f"   window: {WINDOW}")
    print(f"   min_count: {MIN_COUNT}")
    print(f"   epochs: {EPOCHS}")
    print(f"   sg (Skip-gram): {SG}")
    print(f"   negative sampling: {NEGATIVE}")

    # Обучаем модель
    model = Word2Vec(
        sentences=baskets_str,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        epochs=EPOCHS,
        workers=WORKERS,
        sg=SG,
        negative=NEGATIVE,
        seed=SEED
    )

    print(f" Модель обучена!")
    print(f"   Словарь содержит {len(model.wv)} товаров")
    print(f"   (отфильтровано {len(unique_products) - len(model.wv)} редких товаров с count < {MIN_COUNT})")

    print("\n ИЗВЛЕЧЕНИЕ EMBEDDINGS")

    # Получаем embeddings для всех товаров в словаре
    embeddings_dict = {}
    for product_id_str in model.wv.index_to_key:
        product_id = int(product_id_str)
        embeddings_dict[product_id] = model.wv[product_id_str]

    # Конвертируем в numpy array и DataFrame
    product_ids = list(embeddings_dict.keys())
    embeddings_matrix = np.array([embeddings_dict[pid] for pid in product_ids])

    print(f" Извлечено embeddings:")
    print(f"   Shape: {embeddings_matrix.shape}")
    print(f"   (товаров × измерений)")

    # Создаём DataFrame с embeddings
    embeddings_df = pd.DataFrame(
        embeddings_matrix,
        index=product_ids,
        columns=[f'dim_{i}' for i in range(VECTOR_SIZE)]
    )

    # Добавляем метаданные
    embeddings_df['product_id'] = embeddings_df.index
    embeddings_df = embeddings_df.merge(
        unique_products,
        on='product_id',
        how='left'
    )

    print(f"\n DataFrame с embeddings:")
    print(f"   Размер: {embeddings_df.shape}")
    print(f"   Столбцы: {list(embeddings_df.columns[:5])} + метаданные")

    print("\n\n ПЕРВИЧНАЯ ВАЛИДАЦИЯ")

    print("\nТест similarity:")

    # Выбираем несколько товаров для теста
    test_products = [
        ("БАНАНЫ", unique_products[unique_products['name'].str.contains('БАНАНЫ', case=False, na=False)].iloc[0]['product_id'] if len(unique_products[unique_products['name'].str.contains('БАНАНЫ', case=False, na=False)]) > 0 else None),
        ("Молоко", unique_products[unique_products['name'].str.contains('Молоко', case=False, na=False)].iloc[0]['product_id'] if len(unique_products[unique_products['name'].str.contains('Молоко', case=False, na=False)]) > 0 else None),
        ("Хлеб", unique_products[unique_products['name'].str.contains('Хлеб', case=False, na=False)].iloc[0]['product_id'] if len(unique_products[unique_products['name'].str.contains('Хлеб', case=False, na=False)]) > 0 else None),
    ]

    for test_name, test_pid in test_products:
        if test_pid is None or str(test_pid) not in model.wv:
            continue
        
        # Получаем топ-5 похожих товаров
        similar = model.wv.most_similar(str(test_pid), topn=params['w2v']['cnt_similar_items'])
        
        print(f"\n Товар: {test_name} (ID: {test_pid})")
        print(f"   Полное название: {product_metadata[test_pid]['name']}")
        print(f"\n   Топ-5 похожих товаров:")
        
        for i, (similar_id_str, score) in enumerate(similar, 1):
            similar_id = int(similar_id_str)
            similar_name = product_metadata[similar_id]['name']
            similar_cat = product_metadata[similar_id]['art_grp_lvl_0_name']
            print(f"   {i}. [{score:.3f}] {similar_name[:50]} ({similar_cat})")

    sample_test = random.sample(test_baskets, params['w2v']['sample_test'])

    print("Popularity Recall@10:", popularity_recall_random_holdout_word2vec(test_baskets=sample_test, baskets=baskets, random_state=SEED))
    print("Word2Vec Recall@10:", recall_at_k_random_holdout_word2vec(model, sample_test, random_state=SEED))
    
    return embeddings_df

def vae_model(df_products: pd.DataFrame, df_cheques: pd.DataFrame) -> Dict[str, Any]:
    """
    Обучает вариационный автоэнкодер (VAE) для корзин товаров.
    Внутри функции:
    - формируются multi-view представления корзин (товары + категории)
    - создаются train/val/test выборки
    - обучается модель VAE
    - применяется early stopping
    - считается метрика recall@10
    - извлекаются embedding товаров
    Args:
        df_products (pd.DataFrame):
            таблица товаров
        df_cheques (pd.DataFrame):
            таблица чеков
    Returns:
        Dict[str, Any]:
            словарь с результатами:
            {
                "model": обученная модель VAE,
                "config": параметры обучения,
                "item_to_idx": mapping товаров,
                "lvl2_to_idx": mapping категорий,
                "lvl1_to_idx": mapping категорий,
                "lvl0_to_idx": mapping категорий,
                "idx_to_item": обратный mapping
            }
    """

    SEED = params['run_params']['STATE']

    df_final = first_prepair(df_products=df_products, df_cheques=df_cheques)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("СЛОВАРИ ДЛЯ ИЕРАРХИИ")

    # ТОВАРЫ
    items = sorted(df_final['name_clean'].unique())
    item_to_idx = {x: i for i, x in enumerate(items)}

    #УРОВЕНЬ 2
    lvl2 = sorted(df_final['art_grp_lvl_2_name'].unique())
    lvl2_to_idx = {x: i for i, x in enumerate(lvl2)}

    # УРОВЕНЬ 1
    lvl1 = sorted(df_final['art_grp_lvl_1_name'].unique())
    lvl1_to_idx = {x: i for i, x in enumerate(lvl1)}

    # УРОВЕНЬ 0
    lvl0 = sorted(df_final['art_grp_lvl_0_name'].unique())
    lvl0_to_idx = {x: i for i, x in enumerate(lvl0)}

    # СТАТИСТИКА
    print("\nРазмерности:")
    print(f"Items: {len(items)}")
    print(f"Level 2: {len(lvl2)}")
    print(f"Level 1: {len(lvl1)}")
    print(f"Level 0: {len(lvl0)}")

    print("\nПример:")
    sample = df_final.iloc[0]

    print("\nИерархия:")
    print("lvl2:", sample['art_grp_lvl_2_name'])
    print("lvl1:", sample['art_grp_lvl_1_name'])
    print("lvl0:", sample['art_grp_lvl_0_name'])

    print("MULTI-VIEW BASKETS")
    # группируем
    baskets = df_final.groupby('cheque_id').agg({
        'name_clean': list,
        'art_grp_lvl_2_name': list,
        'art_grp_lvl_1_name': list,
        'art_grp_lvl_0_name': list
    }).reset_index()

    print(f"\nВсего корзин: {len(baskets)}")

    # создаём все представления
    item_vectors = []
    lvl2_vectors = []
    lvl1_vectors = []
    lvl0_vectors = []

    for i, row in baskets.iterrows():
        
        if i % 10000 == 0:
            print(f"Обработано: {i}")
        
        item_vectors.append(build_vector(row['name_clean'], item_to_idx))
        lvl2_vectors.append(build_vector(row['art_grp_lvl_2_name'], lvl2_to_idx))
        lvl1_vectors.append(build_vector(row['art_grp_lvl_1_name'], lvl1_to_idx))
        lvl0_vectors.append(build_vector(row['art_grp_lvl_0_name'], lvl0_to_idx))


    # в numpy
    item_vectors = np.array(item_vectors, dtype=np.float32)
    lvl2_vectors = np.array(lvl2_vectors, dtype=np.float32)
    lvl1_vectors = np.array(lvl1_vectors, dtype=np.float32)
    lvl0_vectors = np.array(lvl0_vectors, dtype=np.float32)

    # проверка

    print("\nShapes:")
    print("item:", item_vectors.shape)
    print("lvl2:", lvl2_vectors.shape)
    print("lvl1:", lvl1_vectors.shape)
    print("lvl0:", lvl0_vectors.shape)


    print("TRAIN / VAL / TEST SPLIT")
    cheque_ids = baskets['cheque_id'].values

    # 70 / 15 / 15
    train_idx, temp_idx = train_test_split(
        np.arange(len(baskets)),
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=params['run_params']['test_size_half'],
        random_state=SEED,
        shuffle=True
    )

    # ITEM
    item_train = item_vectors[train_idx]
    item_val = item_vectors[val_idx]
    item_test = item_vectors[test_idx]

    # LVL2
    lvl2_train = lvl2_vectors[train_idx]
    lvl2_val = lvl2_vectors[val_idx]
    lvl2_test = lvl2_vectors[test_idx]

    # LVL1
    lvl1_train = lvl1_vectors[train_idx]
    lvl1_val = lvl1_vectors[val_idx]
    lvl1_test = lvl1_vectors[test_idx]

    # LVL0
    lvl0_train = lvl0_vectors[train_idx]
    lvl0_val = lvl0_vectors[val_idx]
    lvl0_test = lvl0_vectors[test_idx]

    # IDS
    train_ids = cheque_ids[train_idx]
    val_ids = cheque_ids[val_idx]
    test_ids = cheque_ids[test_idx]

    # СТАТИСТИКА
    print("\nРазмеры split:")
    print(f"Train: {len(train_idx):,}")
    print(f"Val:   {len(val_idx):,}")
    print(f"Test:  {len(test_idx):,}")

    print("\nShapes:")
    print("item_train:", item_train.shape)
    print("item_val:  ", item_val.shape)
    print("item_test: ", item_test.shape)

    print("lvl2_train:", lvl2_train.shape)
    print("lvl2_val:  ", lvl2_val.shape)
    print("lvl2_test: ", lvl2_test.shape)

    print("lvl1_train:", lvl1_train.shape)
    print("lvl1_val:  ", lvl1_val.shape)
    print("lvl1_test: ", lvl1_test.shape)

    print("lvl0_train:", lvl0_train.shape)
    print("lvl0_val:  ", lvl0_val.shape)
    print("lvl0_test: ", lvl0_test.shape)

    # SANITY CHECK
    print("\nПример cheque_id:")
    print("train:", train_ids[:3])
    print("val:  ", val_ids[:3])
    print("test: ", test_ids[:3])

    # создаём datasets
    train_dataset = MultiViewDataset(
        item_train, lvl2_train, lvl1_train, lvl0_train
    )

    val_dataset = MultiViewDataset(
        item_val, lvl2_val, lvl1_val, lvl0_val
    )

    print("Dataset создан")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # проверка
    sample = train_dataset[0]

    print("\nSample shapes:")
    for k, v in sample.items():

        print(k, v.shape)

    print("ПОДГОТОВКА КОНТЕКСТА ДЛЯ EVALUATION VAE")

    # МАППИНГ ТОВАР - КАТЕГОРИИ
    item_meta = (
        df_final[['name_clean', 'art_grp_lvl_2_name', 'art_grp_lvl_1_name', 'art_grp_lvl_0_name']]
        .drop_duplicates(subset=['name_clean'])
        .reset_index(drop=True)
    )

    item_to_lvl2 = dict(zip(item_meta['name_clean'], item_meta['art_grp_lvl_2_name']))
    item_to_lvl1 = dict(zip(item_meta['name_clean'], item_meta['art_grp_lvl_1_name']))
    item_to_lvl0 = dict(zip(item_meta['name_clean'], item_meta['art_grp_lvl_0_name']))

    print(f"Маппинги созданы для {len(item_to_lvl2):,} товаров")

    # ИЗ СПИСКА ТОВАРОВ СОБРАТЬ 4 ВЕКТОРА
    build_multiview_from_items_dict = {
        'item_to_idx': item_to_idx, 'lvl2_to_idx': lvl2_to_idx, 
        'lvl1_to_idx': lvl1_to_idx, 'lvl0_to_idx': lvl0_to_idx,
        'item_to_lvl2': item_to_lvl2, 'item_to_lvl1': item_to_lvl1, 
        'item_to_lvl0': item_to_lvl0, 'device': device
    }

    # СОБЕРЁМ TEST-КОРЗИНЫ В ВИДЕ СПИСКОВ ТОВАРОВ
    test_baskets_df = baskets.iloc[test_idx].copy()
    val_baskets_df = baskets.iloc[val_idx].copy()

    print(f"Test корзин: {len(test_baskets_df):,}")

    sample_items = test_baskets_df.iloc[0]['name_clean'][:5]
    print("\nПример товаров из test корзины:")
    print(sample_items)

    sample_x = build_multiview_from_items(sample_items, build_multiview_from_items_dict)

    print("\nShapes sample multiview:")
    print("item:", sample_x[0].shape, "active:", int(sample_x[0].sum()))
    print("lvl2:", sample_x[1].shape, "active:", int(sample_x[1].sum()))
    print("lvl1:", sample_x[2].shape, "active:", int(sample_x[2].sum()))
    print("lvl0:", sample_x[3].shape, "active:", int(sample_x[3].sum()))

    train_dataset = MultiViewDataset(item_train, lvl2_train, lvl1_train, lvl0_train)
    val_dataset = MultiViewDataset(item_val, lvl2_val, lvl1_val, lvl0_val)

    batch_size = params['vae']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_items = item_train.shape[1]
    n_lvl2 = lvl2_train.shape[1]
    n_lvl1 = lvl1_train.shape[1]
    n_lvl0 = lvl0_train.shape[1]


    model = VAE(n_items, n_lvl2, n_lvl1, n_lvl0).to(device)

    lr = float(params['vae']['lr'])
    n_epochs = params['vae']['n_epochs']
    beta = params['vae']['beta']
    train_mask_p = float(params['vae']['train_mask_p'])
    patience = int(params['vae']['patience'])
    min_delta = float(params['vae']['min_delta'])
    counter = int(params['vae']['counter'])
    best_val_recall = float(params['vae']['best_val_recall'])
    best_epoch = int(params['vae']['best_epoch'])

    # TRAIN SETUP
    opt = optim.Adam(model.parameters(), lr=lr)

    n_epochs = n_epochs
    beta = beta
    train_mask_p = train_mask_p

    patience = patience
    min_delta = min_delta
    counter = counter

    best_val_recall = best_val_recall
    best_epoch = best_epoch
    best_state = None

    history = {
        'train_loss': [],
        'train_bce': [],
        'train_kl': [],
        'val_loss': [],
        'val_bce': [],
        'val_kl': [],
        'val_recall@10': [],
    }

    # TRAIN LOOP

    for epoch in range(1, n_epochs + 1):
        model.train()

        train_total_loss = 0.0
        train_total_bce = 0.0
        train_total_kl = 0.0
        train_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            bs = batch['item'].size(0)

            masked = {k: v.clone() for k, v in batch.items()}
            masked['item'] = apply_train_mask(batch['item'], p=train_mask_p)

            recon, mu, logvar = model(masked, sample=True)
            loss, bce, kl = loss_fn(recon, batch, mu, logvar, beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total_loss += loss.item()
            train_total_bce += bce.item()
            train_total_kl += kl.item()
            train_n += bs

            pbar.set_postfix({
                "loss/batch": f"{loss.item()/bs:.2f}",
                "bce/batch": f"{bce.item()/bs:.2f}",
                "kl/batch": f"{kl.item()/bs:.2f}",
                "beta": f"{beta:.2f}",
            })

        train_loss = train_total_loss / train_n
        train_bce  = train_total_bce / train_n
        train_kl   = train_total_kl / train_n

        # VALIDATION LOSS
        model.eval()

        val_total_loss = 0.0
        val_total_bce = 0.0
        val_total_kl = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                bs = batch['item'].size(0)

                recon, mu, logvar = model(batch, sample=False)
                loss, bce, kl = loss_fn(recon, batch, mu, logvar, beta)

                val_total_loss += loss.item()
                val_total_bce += bce.item()
                val_total_kl += kl.item()
                val_n += bs

        val_loss = val_total_loss / val_n
        val_bce  = val_total_bce / val_n
        val_kl   = val_total_kl / val_n

        # VALIDATION RECALL
        val_recall = recall_vae(
            model=model,
            baskets=val_baskets_df, 
            build_multiview_from_items_dict=build_multiview_from_items_dict,
            n=min(params['vae']['sample_test'], len(val_baskets_df)),
            seed=SEED
        )

        history['train_loss'].append(train_loss)
        history['train_bce'].append(train_bce)
        history['train_kl'].append(train_kl)
        history['val_loss'].append(val_loss)
        history['val_bce'].append(val_bce)
        history['val_kl'].append(val_kl)
        history['val_recall@10'].append(val_recall)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_bce: {train_bce:.4f} | "
            f"train_kl: {train_kl:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_bce: {val_bce:.4f} | "
            f"val_kl: {val_kl:.4f} | "
            f"val_recall@10: {val_recall:.4f}"
        )

        # EARLY STOPPING BY VALIDATION RECALL
        if val_recall > best_val_recall + min_delta:
            best_val_recall = val_recall
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
            print(f"  - new best model saved (epoch {epoch}, val_recall@10={val_recall:.4f})")
        else:
            counter += 1
            print(f"  - no recall improvement, early stopping counter: {counter}/{patience}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # RESTORE BEST MODEL
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best model restored from epoch {best_epoch} with val_recall@10={best_val_recall:.4f}")
    else:
        print("Warning: best_state is None, model was not restored.")


    idx_to_item = {v: k for k, v in item_to_idx.items()}
        
    rec = recall_vae(model=model, baskets=test_baskets_df, 
                    build_multiview_from_items_dict=build_multiview_from_items_dict, n=len(test_baskets_df))
    
    config = {
        "latent_dim": 64,
        "beta": beta,
        "mask_p": train_mask_p,
        "best_epoch": best_epoch,
        "test_recall": rec
    }
    
    print(f'Финальная метрика на тестовой выборке: {rec:.4f}')

    vae_results = {
        'model': model,
        'config': config,
        'item_to_idx': item_to_idx,
        'lvl2_to_idx': lvl2_to_idx,
        'lvl1_to_idx': lvl1_to_idx,
        'lvl0_to_idx': lvl0_to_idx,
        'idx_to_item': idx_to_item
    }

    return vae_results

def easy_model(df_products: pd.DataFrame, df_cheques: pd.DataFrame) -> Dict[str, Any]:
    """
    Обучает модель EASE для поиска связей между товарами.
    Внутри функции:
    - формируются корзины
    - строится sparse матрица "корзины × товары"
    - считается матрица co-occurrence (X^T X)
    - применяется регуляризация
    - вычисляются веса модели EASE
    - считается метрика recall@10
    Args:
        df_products (pd.DataFrame):
            таблица товаров

        df_cheques (pd.DataFrame):
            таблица чеков
    Returns:
        Dict[str, Any]:
            словарь с результатами:
            {
                "B_matrix": матрица весов EASE,
                "product_to_idx": mapping товаров в индексы,
                "idx_to_product": обратный mapping,
                "config": параметры модели и метрика
            }
    """
    df_final = first_prepair(df_products=df_products, df_cheques=df_cheques)

    HIER_COLS = params['run_params']['HIER_COLS']
    SEED = params['run_params']['STATE']

    # Уникальные товары по name потому что clean может склеить разные товары в один
    unique_products = df_final[['name', 'name_clean'] + HIER_COLS + ['art_grp_full_name']].drop_duplicates(subset=['name'])

    # Создаём ID для каждого уникального товара
    unique_products = unique_products.reset_index(drop=True)
    unique_products['product_id'] = unique_products.index
    print(f" Уникальных товаров: {len(unique_products):,}")

    # Маппинг: name - product_id
    name_to_id = dict(zip(unique_products['name'], unique_products['product_id']))

    # Добавляем product_id в основной датафрейм
    df_final['product_id'] = df_final['name'].map(name_to_id)

    # уникальные товары
    unique_products = sorted(df_final['product_id'].unique())

    # mapping
    product_to_idx = {p: i for i, p in enumerate(unique_products)}
    idx_to_product = {i: p for p, i in product_to_idx.items()}

    n_items = len(unique_products)

    print(f"\nВсего товаров: {n_items}")

    print("СОЗДАНИЕ КОРЗИН")

    baskets = df_final.groupby('cheque_id')['product_id'].apply(list).reset_index()

    print(f"\nВсего корзин: {len(baskets)}")

    print("\nПример корзины:")
    print(baskets.iloc[0])

    print("SPARSE MATRIX")

    rows = []
    cols = []

    for i, row in baskets.iterrows():
        
        if i % 10000 == 0:
            print(f"Обработано: {i}")
        
        items = row['product_id']
        
        for item in items:
            if item in product_to_idx:
                rows.append(i)
                cols.append(product_to_idx[item])

    data = np.ones(len(rows), dtype=np.float32)

    X = csr_matrix((data, (rows, cols)), shape=(len(baskets), n_items))

    print("\nMatrix shape:")
    print(X.shape)

    print("\nNNZ (ненулевых элементов):")
    print(X.nnz)

    print("\nПлотность:")
    print(X.nnz / (X.shape[0] * X.shape[1]))

    print("TRAIN / TEST SPLIT")

    train_idx, test_idx = train_test_split(
        np.arange(X.shape[0]),
        test_size=TEST_SIZE,
        random_state=SEED
    )

    X_train = X[train_idx]
    X_test  = X[test_idx]

    print(f"Train: {X_train.shape}")
    print(f"Test:  {X_test.shape}")

    lambda_ = params['easy']['best_lambda']

    print(f"lambda = {lambda_}")

    # G = X^T X
    G = (X_train.T @ X_train).toarray().astype(np.float32)

    # регуляризация
    diag_idx = np.diag_indices(G.shape[0])
    G[diag_idx] += lambda_

    # inverse
    P = np.linalg.inv(G)

    # EASE weights
    B = P / (-np.diag(P))
    B[diag_idx] = 0.0

    ease_recall = recall_ease(B, X_test, n_samples=params['easy']['sample_test'])
    print(f"EASE Recall@10: {ease_recall}")

    config = {
        "lambda": lambda_,
        "recall@10": ease_recall
    }

    result = {
        'B_matrix': B,
        'product_to_idx': product_to_idx,
        'idx_to_product': idx_to_product,
        'config': config
        }
    return result

def hybrid_model(df_products: pd.DataFrame, df_cheques: pd.DataFrame) -> Dict[str, Any]:
    """
    Строит гибридную модель на основе EASE и VAE и возвращает все необходимые
    компоненты для дальнейшего использования в качестве генератора признаков.

    Внутри функции выполняются следующие шаги:
    1. Подготовка данных и построение единого пространства товаров.
    2. Обучение модели EASE (линейной модели связей между товарами).
    3. Обучение модели VAE (получение латентных представлений товаров).
    4. Построение признаков для задачи ранжирования.
    5. Обучение модели reranking (логистическая регрессия), которая объединяет
       сигналы EASE и VAE.
    6. Подбор коэффициента объединения скорингов.

    В результате функция возвращает набор объектов, которые позволяют:
    - получать скоринги товаров для заданного контекста корзины,
    - извлекать числовые признаки для отдельных SKU,
    - использовать модель как источник фичей в других задачах.
    Возвращаемое значение представляет собой словарь со следующими элементами:
    "B":
        Матрица весов EASE размерности (n_items, n_items).
        Для каждого товара содержит вектор его связей с другими товарами.
        Может использоваться как источник поведенческих признаков или
        для построения рекомендаций.
    "vae_item_emb":
        Массив размерности (n_items, embedding_dim), содержащий embedding
        каждого товара из модели VAE. Представляет латентное пространство
        товаров и может использоваться как универсальный признак.
    "product_to_idx":
        Словарь, сопоставляющий product_id индексу в матрице EASE.
    "idx_to_product":
        Обратный словарь, позволяющий по индексу получить product_id.
    "ease_to_vae":
        Словарь, сопоставляющий индексы товаров из пространства EASE
        индексам в пространстве VAE. Нужен для согласования двух моделей.
    "build_multiview_from_items_dict":
        Служебный словарь с маппингами, используемый для формирования
        входа в модель VAE при инференсе.
    "pid_to_name":
        Словарь product_id -> name_clean, используется для восстановления
        названий товаров и построения входов для VAE.
    "scaler":
        Объект нормализации признаков (StandardScaler), обученный на данных
        для reranker. Используется при инференсе.
    "reranker":
        Обученная модель логистической регрессии, выполняющая финальное
        ранжирование кандидатов на основе признаков из EASE и VAE.
    "hybrid_scores_fn":
        Функция, принимающая на вход список индексов товаров (контекст корзины)
        и возвращающая кандидатов и их итоговые скоры.
        Используется для получения рекомендаций или скорингов в продакшене.
    Данный набор объектов может быть использован как единый feature engine.
    На его основе можно строить:
    - модели рекомендаций,
    - модели аплифта,
    - модели классификации и прогнозирования,
    так как он предоставляет как поведенческие признаки (EASE),
    так и латентные представления товаров (VAE).
    """
    
    # загружаем датасет
    df_final = first_prepair(df_products=df_products, df_cheques=df_cheques)

    # прогоним модели vae_model, easy_model
    result_vae = vae_model(df_products=df_products, df_cheques=df_cheques)
    result_easy = easy_model(df_products=df_products, df_cheques=df_cheques)

    B = result_easy['B_matrix']
    product_to_idx = result_easy['product_to_idx']
    idx_to_product = result_easy['idx_to_product']

    item_to_idx_vae = result_vae['item_to_idx']
    lvl2_to_idx = result_vae['lvl2_to_idx']
    lvl1_to_idx = result_vae['lvl1_to_idx']
    lvl0_to_idx = result_vae['lvl0_to_idx']
    idx_to_item_vae = result_vae['idx_to_item']
    model = result_vae['model']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    HIER_COLS = params['run_params']['HIER_COLS']
    SEED = params['run_params']['STATE']

    # Уникальные товары по name потому что clean может склеить разные товары в один
    unique_products = df_final[['name', 'name_clean'] + HIER_COLS + ['art_grp_full_name']].drop_duplicates(subset=['name'])

    # Создаём ID для каждого уникального товара
    unique_products = unique_products.reset_index(drop=True)
    unique_products['product_id'] = unique_products.index

    # Маппинг: name - product_id
    name_to_id = dict(zip(unique_products['name'], unique_products['product_id']))

    # Добавляем product_id в основной датафрейм
    df_final['product_id'] = df_final['name'].map(name_to_id)


    baskets = df_final.groupby('cheque_id')['product_id'].apply(list).reset_index()

    rows, cols = [], []

    for i, row in baskets.iterrows():
        for pid in row['product_id']:
            if pid in product_to_idx:
                rows.append(i)
                cols.append(product_to_idx[pid])

    data = np.ones(len(rows), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(len(baskets), len(product_to_idx)))

    train_idx, test_idx = train_test_split(
        np.arange(X.shape[0]),
        test_size=TEST_SIZE,
        random_state=SEED
    )

    X_train = X[train_idx]
    X_test  = X[test_idx]

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # SPACE MAPPINGS

    meta = df_final[['product_id', 'name_clean',
                    'art_grp_lvl_2_name',
                    'art_grp_lvl_1_name',
                    'art_grp_lvl_0_name']].drop_duplicates()

    pid_to_name = dict(zip(meta['product_id'], meta['name_clean']))
    name_to_lvl2 = dict(zip(meta['name_clean'], meta['art_grp_lvl_2_name']))
    name_to_lvl1 = dict(zip(meta['name_clean'], meta['art_grp_lvl_1_name']))
    name_to_lvl0 = dict(zip(meta['name_clean'], meta['art_grp_lvl_0_name']))


    ease_to_vae = {}
    for ease_idx, pid in idx_to_product.items():
        if pid in pid_to_name:
            name = pid_to_name[pid]
            if name in item_to_idx_vae:
                ease_to_vae[ease_idx] = item_to_idx_vae[name]

    print("Matched items:", len(ease_to_vae), "of", len(idx_to_product))

    build_multiview_from_items_dict = {
            'item_to_idx': item_to_idx_vae, 'lvl2_to_idx': lvl2_to_idx, 
            'lvl1_to_idx': lvl1_to_idx, 'lvl0_to_idx': lvl0_to_idx,
            'item_to_lvl2': name_to_lvl2, 'item_to_lvl1': name_to_lvl1, 
            'item_to_lvl0': name_to_lvl0, 'device': device
        }

    vae_item_emb = model.item_dec.weight.detach().cpu().numpy()
    vae_item_emb = vae_item_emb / (np.linalg.norm(vae_item_emb, axis=1, keepdims=True) + 1e-8)

    X_rank, y_rank, group = build_ranking_dataset(
        X_source=X_train, 
        ease_to_vae=ease_to_vae, 
        vae_item_emb=vae_item_emb, 
        build_multiview_from_items_dict=build_multiview_from_items_dict,
        B=B, idx_to_product=idx_to_product, pid_to_name=pid_to_name, model=model,
        n_samples=params['hybrid']['n_samples'],
        topk_candidates=params['hybrid']['topk_candidates'],
        seed=SEED
    )

    print("X_rank:", X_rank.shape)
    print("y_rank:", y_rank.shape)
    print("groups:", len(group))
    print("positive rate:", y_rank.mean())

    print("TRAIN FINAL LOGISTIC RERANKER")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rank)

    reranker = LogisticRegression(
        C=params['hybrid']['log_reg_C'],
        max_iter=params['hybrid']['log_reg_max_iter'],
        class_weight="balanced",
        solver="lbfgs"
    )

    reranker.fit(X_scaled, y_rank)

    best_alpha = params['hybrid']['best_alpha']
    best_recall = recall_hybrid_model(
        X_test, 
        hybrid_fn=lambda x: hybrid_scores(context=x, 
                                        build_multiview_from_items_dict=build_multiview_from_items_dict,
                                        ease_to_vae=ease_to_vae,
                                        vae_item_emb=vae_item_emb,
                                        B=B, idx_to_product=idx_to_product, 
                                        pid_to_name=pid_to_name, 
                                        model=model,
                                        scaler=scaler,
                                        reranker=reranker,
                                        topk_candidates=params['hybrid']['topk_candidates'],
                                        alpha=best_alpha)
    )

    print(f'Финальный Recall@10 гибрида = {best_recall:.4f}')

    # ПРИМЕР: ПРИЗНАКИ ДЛЯ НЕСКОЛЬКИХ SKU ИЗ ЧЕКА

    sample_basket = df_final.groupby('cheque_id')['product_id'].apply(list).iloc[0]
    sample_items = sample_basket[:3]

    print('Пример признаков в виде векторов связей между продуктами (easy) и эмбеддингов (vae)')
    for pid in sample_items:
        feat = get_sku_features(
                                pid, 
                                product_to_idx=product_to_idx, 
                                idx_to_product=idx_to_product, 
                                ease_to_vae=ease_to_vae, 
                                B=B, 
                                vae_item_emb=vae_item_emb,
                                topn_ease=params['hybrid']['topn_ease']
                                )
        print("SKU:", feat["product_id"])
        print("Top EASE relations:")
        for rel_pid, w in feat["ease_top_related"]:
            print(f"  {rel_pid}: {w:.4f}")
        print("VAE embedding head:")
        print(feat["vae_embedding"][:10] if feat["vae_embedding"] is not None else None)

    print()
    print()
    print()

    all_lists_basket = df_final.groupby('cheque_id')['product_id'].apply(list).iloc[random.randint(1,50)]
    sample_items = all_lists_basket[:1]

    id_to_name = dict(zip(df_final['product_id'], df_final['name_clean']))

    print('Более внятный пример, если мы хотим увидеть не айди продукта, а само название')
    for pid in sample_items:
        feat = get_sku_features(
                                pid, 
                                product_to_idx=product_to_idx, 
                                idx_to_product=idx_to_product, 
                                ease_to_vae=ease_to_vae, 
                                B=B, 
                                vae_item_emb=vae_item_emb,
                                topn_ease=params['hybrid']['topn_ease']
                                )
        
        print("SKU:", id_to_name.get(feat["product_id"], feat["product_id"]))
        
        print("\nTop связанные товары (EASE):")
        for rel_pid, w in feat["ease_top_related"]:
            print(f"  {id_to_name.get(rel_pid, rel_pid)} - {w:.4f}")

    return {
            # модели 
            "B": B,
            "vae_item_emb": vae_item_emb,

            # маппинги 
            "product_to_idx": product_to_idx,
            "idx_to_product": idx_to_product,
            "ease_to_vae": ease_to_vae,

            # для VAE
            "build_multiview_from_items_dict": build_multiview_from_items_dict,
            "pid_to_name": pid_to_name,

            # модели ранжирования 
            "scaler": scaler,
            "reranker": reranker,

            # функция скоринга
            "hybrid_scores_fn": lambda context: hybrid_scores(
                context=context,
                build_multiview_from_items_dict=build_multiview_from_items_dict,
                ease_to_vae=ease_to_vae,
                vae_item_emb=vae_item_emb,
                B=B,
                idx_to_product=idx_to_product,
                pid_to_name=pid_to_name,
                model=model,
                scaler=scaler,
                reranker=reranker,
                topk_candidates=params['hybrid']['topk_candidates'],
                alpha=best_alpha
            )
        }  
