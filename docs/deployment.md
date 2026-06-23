# Deployment

## Local Streamlit

```bash
streamlit run app/app.py
```

## Docker Compose

```bash
docker compose up --build
```

The app runs on port `8501`.

## Notes

Keep large datasets, embedding matrices, and generated model artifacts outside Git. Store only small sample files or reproducible configuration in the repository.
