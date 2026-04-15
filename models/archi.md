models/
├── als/
│   ├── user_ids.npy
│   ├── user_vectors.npy
│   ├── item_ids.npy
│   ├── item_vectors.npy
│   └── metadata.json      # rank, date d'entraînement, RMSE, dataset
├── svdpp/                 # quand on l'exportera
│   ├── user_factors.npy
│   ├── ...
│   └── metadata.json
└── shared/                # commun aux deux modèles
    ├── movies.csv
    ├── links.csv
    └── movie_id_to_idx.json