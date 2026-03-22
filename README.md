bot-detection-kg/
│
├── README.md
├── requirements.txt
├── config/
│   ├── base.yaml
│   ├── twibot22.yaml
│   └── training.yaml
│
├── data/
│   ├── raw/
│   │   └── twibot22/
│   │       ├── user.json
│   │       ├── tweet.json
│   │       ├── edge.csv
│   │       └── label.csv
│   │
│   ├── interim/
│   │   ├── processed_users.pkl
│   │   ├── processed_tweets.pkl
│   │   └── extracted_entities.pkl
│   │
│   └── processed/
│       └── hetero_graph.pt
│
├── src/
│   ├── data/
│   │   ├── loader.py              # load raw dataset
│   │   ├── preprocessor.py        # clean + normalize data
│   │   ├── entity_extractor.py    # hashtags, URLs from tweets
│   │   ├── graph_builder.py       # build HeteroData
│   │   └── splits.py              # train/val/test split
│   │
│   ├── features/
│   │   ├── user_features.py
│   │   ├── tweet_features.py
│   │   ├── text_encoder.py        # BERT / TF-IDF
│   │   └── feature_utils.py
│   │
│   ├── models/
│   │   ├── hetero_gnn.py          # RGCN / HAN / HGT
│   │   ├── layers.py
│   │   └── utils.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluator.py
│   │   └── loss.py
│   │
│   ├── explainability/
│   │   ├── explainer.py           # GNNExplainer / PGExplainer
│   │   ├── subgraph_extractor.py
│   │   └── visualization.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── seed.py
│       └── config.py
│
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_analysis.ipynb
│   └── graph_visualization.ipynb
│
├── scripts/
│   ├── preprocess.sh
│   ├── train.sh
│   └── run_pipeline.sh
│
└── outputs/
    ├── logs/
    ├── checkpoints/
    └── figures/

users_df   → structured user table
tweets_df  → structured tweet table
edges_df   → clean edge list
labels_df  → clean label mapping