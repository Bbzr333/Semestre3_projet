"""
Découpage stratifié des données en train/val/test
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import json

class DataSplitter:
    """
    Gère le découpage et la sauvegarde des datasets
    """
    
    def __init__(self, test_size=0.15, val_size=0.15, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split_data(self, df, target_col='type_jdm'):
        """
        Découpe les données en train/val/test de manière stratifiée
        """
        # Premier split : train+val / test
        train_val, test = train_test_split(
            df, 
            test_size=self.test_size,
            stratify=df[target_col],
            random_state=self.random_state
        )
        
        # Second split : train / val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val[target_col],
            random_state=self.random_state
        )
        
        # Stats
        stats = {
            'total_samples': len(df),
            'train_samples': len(train),
            'val_samples': len(val),
            'test_samples': len(test),
            'train_ratio': len(train) / len(df),
            'val_ratio': len(val) / len(df),
            'test_ratio': len(test) / len(df),
            'num_classes': df[target_col].nunique(),
            'class_distribution': df[target_col].value_counts().to_dict()
        }
        
        return train, val, test, stats
    
    def save_splits(self, train, val, test, stats, output_dir='data/processed'):
        """
        Sauvegarde les splits et les statistiques
        """
        train.to_csv(f'{output_dir}/train.csv', index=False)
        val.to_csv(f'{output_dir}/val.csv', index=False)
        test.to_csv(f'{output_dir}/test.csv', index=False)
        
        with open(f'{output_dir}/split_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Splits sauvegardés dans {output_dir}/")
        print(f"📊 Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")