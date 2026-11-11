"""Train baseline: TF-IDF (char n-gram) + Logistic Regression.
Binary: normal=0, spam=1, taken directly from the xlsx 'label' column plus optional supplementary normal examples."""
import os, re, joblib, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

EXTRA_NORMAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extra_normal.csv')
COLLECTED_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'collected_data_clean.csv')
EXTRA_SAMPLE_WEIGHT = 0.55
COLLECTED_NORMAL_WEIGHT = 0.20
COLLECTED_SPAM_WEIGHT = 1.50

def clean(t):
    t = re.sub(r'[\U00010000-\U0010ffff☀-➿️]', ' ', str(t))  # drop emoji (noise)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)                                    # soooo -> soo
    return re.sub(r'\s+', ' ', t).strip().lower()

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    d = pd.read_excel(os.path.join(base_dir, 'data.xlsx'))
    d['y'] = (d.label == 'spam').astype(int)
    d['clean'] = d.text.map(clean)
    d = d.drop_duplicates('clean').reset_index(drop=True)
    # messages are generated from templates -> group by template, otherwise CV leaks (100% acc)
    d['grp'] = d.clean.str.replace(r'[0-9a-z\W_]+', '', regex=True).str[:14]
    print('Base dataset counts:', d.y.value_counts().sort_index().to_dict())

    mk = lambda: make_pipeline(
        TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=2000, class_weight='balanced', C=5))

    cv = StratifiedGroupKFold(3, shuffle=True, random_state=0)
    names = ['normal', 'spam']
    pred = cross_val_predict(mk(), d.clean, d.y, cv=cv, groups=d.grp)
    print('\n=== Benchmark 3-Fold Group CV (data.xlsx only) ===')
    print(classification_report(d.y, pred, target_names=names, digits=3))
    print(confusion_matrix(d.y, pred))

    sup_parts, sup_weights = [], []
    existing_clean = set(d['clean'])

    if os.path.exists(EXTRA_NORMAL_PATH):
        extra = pd.read_csv(EXTRA_NORMAL_PATH, encoding='utf-8-sig')
        extra['clean'] = extra.text.map(clean)
        extra['y'] = 0
        for c, y in zip(extra['clean'], extra['y']):
            if c and c not in existing_clean:
                existing_clean.add(c)
                sup_parts.append((c, y))
                sup_weights.append(EXTRA_SAMPLE_WEIGHT)

    if os.path.exists(COLLECTED_DATA_PATH):
        col = pd.read_csv(COLLECTED_DATA_PATH, encoding='utf-8-sig')
        col['clean'] = col.text.map(clean)
        col['y'] = (col.label.str.strip().str.lower() == 'spam').astype(int)
        for c, y in zip(col['clean'], col['y']):
            if c and c not in existing_clean:
                existing_clean.add(c)
                sup_parts.append((c, y))
                sup_weights.append(COLLECTED_SPAM_WEIGHT if y == 1 else COLLECTED_NORMAL_WEIGHT)

    if sup_parts:
        sup_df = pd.DataFrame(sup_parts, columns=['clean', 'y'])
        sup_w = np.array(sup_weights, dtype=float)
        print(f'\nLoaded {len(sup_df)} supplementary examples (normal={(sup_df.y==0).sum()}, spam={(sup_df.y==1).sum()})')

        oof = np.zeros(len(d), dtype=int)
        for tr_idx, va_idx in cv.split(d.clean, d.y, d.grp):
            tr_df = pd.concat([d.iloc[tr_idx][['clean', 'y']], sup_df], ignore_index=True)
            sw_fold = np.concatenate([np.ones(len(tr_idx)), sup_w])
            m_fold = mk().fit(tr_df.clean, tr_df.y, logisticregression__sample_weight=sw_fold)
            oof[va_idx] = m_fold.predict(d.iloc[va_idx].clean)

        print('=== Out-of-Fold CV on data.xlsx (with supplementary data in train folds) ===')
        print(classification_report(d.y, oof, target_names=names, digits=3))
        print(confusion_matrix(d.y, oof))

        full = pd.concat([d[['clean', 'y']], sup_df], ignore_index=True)
        sw = np.concatenate([np.ones(len(d)), sup_w])
        final_model = mk().fit(full.clean, full.y, logisticregression__sample_weight=sw)
    else:
        final_model = mk().fit(d.clean, d.y)

    joblib.dump(final_model, os.path.join(base_dir, 'model.joblib'))
    print('\nSaved updated model to model.joblib')
