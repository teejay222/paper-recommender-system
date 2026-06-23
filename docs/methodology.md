# Methodology

The recommender can be developed in phases:

1. Collect paper metadata from sources such as OpenReview, Semantic Scholar, or arXiv.
2. Normalize metadata into a consistent schema with title, abstract, authors, venue, year, and keywords.
3. Generate text embeddings from title and abstract fields.
4. Rank candidate papers with cosine similarity or a learned similarity model.
5. Evaluate recommendations using annotated relevance labels.
6. Compare experiments across embedding models, query strategies, and ranking methods.

Suggested metrics include precision at K, recall at K, mean average precision, and nDCG.
