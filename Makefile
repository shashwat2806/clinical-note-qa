.PHONY: run test eval index clean help

help:
	@echo "Available commands:"
	@echo "  make run    - Start the Streamlit chat UI"
	@echo "  make test   - Run all unit tests"
	@echo "  make eval   - Run the evaluation benchmark"
	@echo "  make index  - Rebuild the vector index from clinical notes"
	@echo "  make clean  - Remove cache files and the vector store"

run:
	cd src && streamlit run app.py

test:
	python -m pytest tests/ -v

eval:
	cd src && python evaluate.py

index:
	cd src && python vectorstore.py

clean:
	rm -rf src/__pycache__ src/chroma_store tests/__pycache__ .pytest_cache