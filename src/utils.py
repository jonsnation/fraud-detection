def frequency_encode(df, columns, max_unique=500):
    for col in columns:
        if df[col].nunique(dropna=True) <= max_unique:
            freq = df[col].value_counts(dropna=False) / len(df)
            df[col + '_freq'] = df[col].map(freq)
    return df

def add_missing_flags(df, prefix='id_'):
    for col in df.columns:
        if col.startswith(prefix) and df[col].dtype in ['float64', 'int64']:
            if df[col].isnull().mean() < 0.5:
                df[col + '_missing'] = df[col].isnull().astype(int)
    return df
